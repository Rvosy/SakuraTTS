"""Exercise the sticker allowlist at the outgoing message boundary."""
import base64
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from maibot_sdk.context import PluginContext

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('emoji_policy', ROOT / 'plugin.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ASSETS = ROOT.parent / 'maibot_navi_expressions' / 'assets'


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = module.create_plugin()
        self.plugin._set_context(PluginContext('rvosy.emoji-send-policy'))

    async def check(self, components, expected):
        result = await self.plugin.stop_outbound_emoji(message={'raw_message': components})
        self.assertEqual(result['action'], expected)

    async def test_only_three_original_assets_allowed_as_stickers(self):
        assets = list(ASSETS.glob('*.png'))
        self.assertEqual(len(assets), 3)
        for image in assets:
            await self.check([{'type': 'emoji', 'binary_data_base64': base64.b64encode(image.read_bytes()).decode()}], 'continue')

    async def test_unknown_forged_or_missing_payload_blocked(self):
        for data in ['', '???', None, base64.b64encode(b'other picture').decode()]:
            await self.check([{'type': 'emoji', 'data': 'N.A.V.I. 的表情包：开心',
                               'hash': next(iter(module.ALLOWED_SHA256)), 'binary_data_base64': data}], 'abort')

    async def test_nested_forward_cannot_hide_other_sticker(self):
        image = next(ASSETS.glob('*.png'))
        allowed = {'type': 'emoji', 'binary_data_base64': base64.b64encode(image.read_bytes()).decode()}
        def forward(items): return {'type': 'forward', 'data': [{'content': items}]}
        await self.check([forward([forward([allowed])])], 'continue')
        await self.check([forward([allowed, forward([{'type': 'emoji'}])])], 'abort')

    async def test_text_voice_and_ordinary_image_are_unchanged(self):
        await self.check([{'type': kind, 'data': 'test'} for kind in ['text', 'voice', 'image']], 'continue')

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
