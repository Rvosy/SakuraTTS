"""禁用通用表情库，固定素材由角色表情插件提供。"""
from copy import deepcopy

from maibot_sdk import HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode, HookOrder


class EmojiSendPolicy(MaiBotPlugin):
    async def on_load(self):
        self.ctx.logger.info('通用表情库已禁用，固定表情由角色插件提供')

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
            tool = original
            if name == 'reply':
                tool = deepcopy(original)
                parameters = tool['function']['parameters']
                parameters.get('properties', {}).pop('attach_emoji', None)
                if 'required' in parameters:
                    parameters['required'] = [x for x in parameters['required'] if x != 'attach_emoji']
            definitions.append(tool)
        return {'action': 'continue', 'modified_kwargs': {**kwargs, 'tool_definitions': definitions}}

    @HookHandler('emoji.maisaka.before_select', mode=HookMode.BLOCKING, order=HookOrder.EARLY)
    async def stop_selection(self, **kwargs):
        return {'action': 'abort', 'modified_kwargs': {**kwargs, 'abort_message': '通用表情库已禁用，请使用角色的固定表情工具。'}}


def create_plugin():
    return EmojiSendPolicy()
