"""Lower length-preserving 1D upsampling to Conv plus phase interleaving."""

import numpy as np
import onnx


def lower_conv_transpose_polyphase(model):
    """Pack output phases into channels without materializing inserted zeros.

    Supports the current vocoder's group=1, dilation=1, output_length=L*stride
    transposed convolutions. Unsupported padding is rejected before mutation.
    """
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes, added, removed, changes = [], [], set(), []
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            nodes.append(node)
            continue
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        weight = onnx.numpy_helper.to_array(tensors[node.input[1]])
        strides, pads = attrs.get("strides", [1]), attrs.get("pads", [0, 0])
        if (weight.ndim != 3 or len(strides) != 1 or len(pads) != 2
                or attrs.get("group", 1) != 1 or attrs.get("dilations", [1]) != [1]
                or attrs.get("output_padding", [0]) != [0] or "output_shape" in attrs
                or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"):
            raise ValueError(f"Unsupported polyphase ConvTranspose attributes: {node.name}")
        stride, kernel = strides[0], weight.shape[-1]
        if (stride < 1 or kernel < stride or min(pads) < 0 or sum(pads) != kernel - stride
                or attrs.get("kernel_shape", [kernel]) != [kernel]):
            raise ValueError(f"Polyphase lowering requires output length = input length * stride: {node.name}")

        # y[j*S+r] = sum_i x[i] W[(j-i)*S+r+pad_left].
        # Reverse each residue class for ordinary cross-correlation. Different
        # left pads are represented by zeros in the packed kernel, not input.
        phases = []
        for phase in range(stride):
            residue = (phase + pads[0]) % stride
            indices = list(range(residue, kernel, stride))[::-1]
            left = len(indices) - 1 + (residue - pads[0] - phase) // stride
            right = len(indices) - 1 - left
            if min(left, right) < 0:
                raise ValueError(f"Polyphase lowering requires nonnegative phase padding: {node.name}")
            phases.append((indices, left, right))
        left, right = max(p[1] for p in phases), max(p[2] for p in phases)
        packed_kernel = left + right + 1
        cin, cout = weight.shape[:2]
        packed = np.zeros((cout, stride, cin, packed_kernel), dtype=weight.dtype)
        for phase, (indices, phase_left, _) in enumerate(phases):
            offset = left - phase_left
            packed[:, phase, :, offset:offset+len(indices)] = weight[:, :, indices].transpose(1, 0, 2)
        prefix = node.name + "/polyphase"
        def constant(suffix, value):
            name = prefix + "/" + suffix
            added.append(onnx.numpy_helper.from_array(value, name=name))
            return name
        packed_name = constant("weight", packed.reshape(cout * stride, cin, packed_kernel))
        inputs = [node.input[0], packed_name]
        if len(node.input) == 3:
            bias = onnx.numpy_helper.to_array(tensors[node.input[2]])
            inputs.append(constant("bias", np.repeat(bias, stride)))
            removed.add(node.input[2])
        split_shape = constant("split_shape", np.asarray([0, cout, stride, -1], np.int64))
        output_shape = constant("output_shape", np.asarray([0, cout, -1], np.int64))
        convolved, split, interleaved = (prefix + "/" + name for name in ("convolved", "split", "interleaved"))
        nodes.extend([
            onnx.helper.make_node("Conv", inputs, [convolved], name=prefix+"/Conv",
                kernel_shape=[packed_kernel], pads=[left, right], strides=[1], dilations=[1], group=1),
            onnx.helper.make_node("Reshape", [convolved, split_shape], [split], name=prefix+"/Split"),
            onnx.helper.make_node("Transpose", [split], [interleaved], perm=[0, 1, 3, 2], name=prefix+"/Interleave"),
            onnx.helper.make_node("Reshape", [interleaved, output_shape], list(node.output), name=prefix+"/Output"),
        ])
        removed.add(node.input[1])
        changes.append({"node": node.name, "method": "polyphase", "weight_shape": list(weight.shape),
            "stride": stride, "original_pads": pads, "packed_weight_shape": list(packed.reshape(cout*stride, cin, packed_kernel).shape),
            "conv_pads": [left, right], "phases": [{"kernel_indices": p[0], "left_pad": p[1], "right_pad": p[2]} for p in phases],
            "output_interleave": "(N,Cout,stride,L) -> (N,Cout,L,stride) -> (N,Cout,L*stride)"})
    consumed = {name for node in nodes for name in node.input} | {value.name for value in model.graph.output}
    retained = [tensor for tensor in model.graph.initializer if tensor.name not in removed or tensor.name in consumed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *added])
    return changes
