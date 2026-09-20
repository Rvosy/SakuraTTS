"""Private CPU-only worker for the packaged classic pyopenjtalk frontend."""

import argparse
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).with_name("_worker_bootstrap.py")))["load_package"](Path(__file__).resolve().parent)
from sakuratts.array_protocol import read_message, write_message
from sakuratts.japanese import JapaneseG2P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-directory", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--user-dictionary", type=Path, required=True)
    args = parser.parse_args()
    try:
        module_directory = args.module_directory.resolve(strict=True)
        dictionary = args.dictionary.resolve(strict=True)
        user_dictionary = args.user_dictionary.resolve(strict=True)
        for name in ("char.bin", "matrix.bin", "sys.dic", "unk.dic"):
            if not (dictionary / name).is_file():
                raise FileNotFoundError(dictionary / name)
        os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary)
        sys.path.insert(0, str(module_directory))
        import pyopenjtalk

        if pyopenjtalk.__version__ != "0.3.4":
            raise ValueError("Expected the classic pyopenjtalk 0.3.4 profile")
        if Path(pyopenjtalk.__file__).resolve().parent.parent != module_directory:
            raise ValueError("pyopenjtalk was not loaded from the configured module directory")
        # Explicit existing dictionaries avoid the package's download/build path.
        jtalk = pyopenjtalk.OpenJTalk(dn_mecab=os.fsencode(dictionary), userdic=os.fsencode(user_dictionary))
        adapter = object.__new__(JapaneseG2P)
        adapter._jtalk = jtalk
        adapter.labels = lambda text: jtalk.make_label(jtalk.run_frontend(text))
        write_message(sys.stdout.buffer, {"status": "ready", "implementation": "pyopenjtalk-classic",
            "version": pyopenjtalk.__version__, "python": sys.version,
            "module_directory": str(module_directory), "main_dictionary": str(dictionary),
            "torch_imported": "torch" in sys.modules, "onnxruntime_imported": "onnxruntime" in sys.modules})
        while True:
            request, arrays = read_message(sys.stdin.buffer)
            if arrays:
                raise ValueError("Classic frontend accepts text metadata only")
            if request.get("command") == "close":
                break
            if request.get("command") != "g2p" or not isinstance(request.get("text"), str):
                raise ValueError("Expected a Japanese G2P text request")
            phones = adapter.g2p(request["text"])
            write_message(sys.stdout.buffer, {"status": "ok", "phones": phones})
    except EOFError:
        pass
    except BaseException:
        write_message(sys.stdout.buffer, {"status": "error", "error": traceback.format_exc()})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
