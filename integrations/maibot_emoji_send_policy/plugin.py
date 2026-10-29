"""User-requested outbound emoji policy; independent of character and TTS."""
from copy import deepcopy
import base64
import binascii
import hashlib

from maibot_sdk import HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode, HookOrder


# 校验实际图片字节，不凭名称、说明或调用方提供的 hash 放行。
ALLOWED_SHA256 = {
    '31e7e91426afbabc1ca15c8412955f676f7d1041fcf7fd2c6b03fa24a872a089',  # 无奈认真
    'c8b796b1fae17fb27d75a6c472ce1dbbcf49ba7729aa87bd3bcd1b2ff0e99764',  # 意料之外
    'b93a044aa652e5e8fb68af77b844184f715309d9188a00b7014440f0e8913d4e',  # 开心
}


def contains_disallowed_emoji(components):
    for component in components:
        if component.get('type') == 'emoji':
            try:
                data = base64.b64decode(component.get('binary_data_base64', ''), validate=True)
            except (ValueError, TypeError, binascii.Error):
                return True
            if hashlib.sha256(data).hexdigest() not in ALLOWED_SHA256:
                return True
        if component.get('type') == 'forward':
            if any(contains_disallowed_emoji(node.get('content', [])) for node in component.get('data', [])):
                return True
    return False


class EmojiSendPolicy(MaiBotPlugin):
    async def on_load(self):
        self.ctx.logger.info('表情包发送策略 1.1.0 已加载，仅放行 N.A.V.I. 的三张固定表情')

    async def on_unload(self):
        pass

    async def on_config_update(self, scope, config_data, version):
        pass

    @HookHandler('maisaka.planner.before_request', mode=HookMode.BLOCKING, order=HookOrder.LATE)
    async def hide_emoji_tools(self, **kwargs):
        definitions = []
        for original in kwargs['tool_definitions']:
            name = original.get('function', {}).get('name')
            if name in {'send_emoji', 'show_emoji_list'}:
                continue
            tool = deepcopy(original)
            if name == 'reply':
                parameters = tool['function']['parameters']
                parameters.get('properties', {}).pop('attach_emoji', None)
                if 'required' in parameters:
                    parameters['required'] = [x for x in parameters['required'] if x != 'attach_emoji']
            definitions.append(tool)
        return {'action': 'continue', 'modified_kwargs': {**kwargs, 'tool_definitions': definitions}}

    @HookHandler('emoji.maisaka.before_select', mode=HookMode.BLOCKING, order=HookOrder.EARLY)
    async def stop_selection(self, **kwargs):
        return {'action': 'abort', 'modified_kwargs': {**kwargs, 'abort_message': '通用表情库已禁用，请使用角色的固定表情工具。'}}

    @HookHandler('emoji.maisaka.after_select', mode=HookMode.BLOCKING, order=HookOrder.LATE)
    async def stop_selected_emoji(self, **kwargs):
        return {'action': 'abort', 'modified_kwargs': {**kwargs, 'abort_message': '通用表情库已禁用，请使用角色的固定表情工具。'}}

    @HookHandler('send_service.before_send', mode=HookMode.BLOCKING, order=HookOrder.EARLY)
    async def stop_outbound_emoji(self, **kwargs):
        if contains_disallowed_emoji(kwargs['message'].get('raw_message', [])):
            self.ctx.logger.info('已拦截固定素材之外的表情包')
            return {'action': 'abort'}
        return {'action': 'continue'}


def create_plugin():
    return EmojiSendPolicy()
