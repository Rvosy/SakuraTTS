"""验证通用表情选择入口与固定表情工具的分工。"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from maibot_sdk.context import PluginContext

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('emoji_policy', ROOT / 'plugin.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = module.create_plugin()
        self.plugin._set_context(PluginContext('rvosy.emoji-send-policy'))

    async def test_generic_picker_hidden_and_specific_tools_remain(self):
        tools = [{'function': {'name': name}} for name in ['send_emoji', 'show_emoji_list',
                 'navi_expression_happy', 'navi_expression_serious', 'navi_expression_unexpected']]
        tools.append({'function': {'name': 'reply', 'parameters': {
            'properties': {'attach_emoji': {}, 'text': {}}, 'required': ['attach_emoji', 'text']}}})
        original = deepcopy(tools)
        updated = (await self.plugin.hide_emoji_tools(tool_definitions=tools, session_id='group'))['modified_kwargs']
        self.assertEqual(tools, original)
        self.assertEqual([t['function']['name'] for t in updated['tool_definitions']],
                         ['navi_expression_happy', 'navi_expression_serious', 'navi_expression_unexpected', 'reply'])
        self.assertEqual(updated['tool_definitions'][-1]['function']['parameters'],
                         {'properties': {'text': {}}, 'required': ['text']})
        self.assertEqual((await self.plugin.stop_selection())['action'], 'abort')


if __name__ == '__main__':
    unittest.main()
