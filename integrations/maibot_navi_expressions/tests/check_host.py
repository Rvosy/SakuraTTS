"""Run inside an isolated MaiBot container; all platform and database writes are intercepted."""
import ast
import asyncio
import base64
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

shutil.copytree('/source-config', '/MaiMBot/config', dirs_exist_ok=True)
from maibot_sdk.context import PluginContext, PluginPaths
from src.services import send_service as send
from src.plugin_runtime.capabilities.core import RuntimeCoreCapabilityMixin
from src.plugin_runtime.host.component_registry import ToolEntry, ReplyExtensionEntry
from src.plugin_runtime.host import reply_extensions as extensions
from src.core.tooling import ToolAvailabilityContext
from src.plugin_runtime.runner.manifest_validator import ManifestValidator
from src.common.utils.utils_message import MessageUtils
from src.maisaka.context.message_adapter import build_visible_text_from_sequence


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load('expressions', '/plugin/plugin.py')
# Import the installed adapter encoder without starting its plugin or network services.
for name, directory in [('adapter_check', '/adapter'), ('adapter_check.codecs', '/adapter/codecs'),
                        ('adapter_check.codecs.outbound', '/adapter/codecs/outbound')]:
    package = types.ModuleType(name)
    package.__path__ = [directory]
    sys.modules[name] = package
from adapter_check.codecs.outbound.segment_encoder import QQOutboundSegmentEncoder
encoder = QQOutboundSegmentEncoder()
source = Path('src/maisaka/runtime.py').read_text(encoding='utf-8-sig')
node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
            and n.name == 'append_sent_message_to_chat_history')
namespace = {'logger': send.logger}
exec('from __future__ import annotations\n' + ast.get_source_segment(source, node), namespace)
runtime = types.SimpleNamespace(
    _chat_history=[], _focus_group=None, _quote_counter=Mock(), log_prefix='expression-check',
    _is_focus_mode_active_for_current_chat=lambda: False,
    _schedule_sent_image_recognition=Mock(), _emit_monitor_message_sent=Mock(),
)
runtime.append_sent_message_to_chat_history = types.MethodType(namespace['append_sent_message_to_chat_history'], runtime)
heart = types.ModuleType('src.chat.heart_flow.heartflow_manager')
heart.heartflow_manager = types.SimpleNamespace(heartflow_chat_list={'expression-test': runtime})


class Platform:
    succeeds = True
    captured = []
    async def ensure_send_pipeline_ready(self): pass
    def build_route_key_from_message(self, message):
        return types.SimpleNamespace(platform='qq', account_id='test-bot')
    async def send_message(self, message, route_key, **kwargs):
        segments = await MessageUtils.from_MaiSeq_to_maim_message_segments(message.raw_message)
        self.captured.append(segments)
        encoded = encoder.convert_segments(send.serialize_session_message(message)['raw_message'])
        for original, converted in zip(segments, encoded, strict=True):
            if original.type == 'emoji':
                assert converted['type'] == 'image' and converted['data']['sub_type'] == 1
                assert converted['data']['file'] == 'base64://' + original.data
        return types.SimpleNamespace(has_success=self.succeeds,
            sent_receipts=[types.SimpleNamespace(external_message_id=f'test-image-{len(self.captured)}', metadata={})] if self.succeeds else [],
            failed_receipts=[], route_key=route_key)


async def main():
    validator = ManifestValidator(log_errors=False)
    assert validator.load_from_plugin_path(Path('/plugin')), validator.errors
    stored = []
    async def store(message): stored.append(message.to_db_instance())
    async def hook(name, message, **kwargs):
        if name == 'send_service.after_send':
            await plugin.record_sent_expression(message=send.serialize_session_message(message), sent=kwargs['sent'])
            return types.SimpleNamespace(aborted=False, kwargs=kwargs), message
        return types.SimpleNamespace(aborted=False, kwargs=kwargs), message
    platform = Platform()
    host = RuntimeCoreCapabilityMixin()
    async def rpc(method, plugin_id, payload, **kwargs):
        assert payload['capability'] == 'send.hybrid'
        return await host._cap_send_hybrid(plugin_id, payload['capability'], dict(payload['args'], show_log=False))
    with tempfile.TemporaryDirectory() as temporary:
        plugin = module.create_plugin()
        plugin.set_plugin_config({})
        plugin._set_context(PluginContext('rvosy.navi-expressions', rpc_call=rpc,
            paths=PluginPaths(Path(temporary), Path(temporary)/'runtime')))
        await plugin.on_load()
        for component in plugin.get_components():
            if component['type'] != 'TOOL':
                continue
            entry = ToolEntry(component['name'], component['type'], 'rvosy.navi-expressions', component['metadata'])
            assert entry._get_parameters_schema()['properties']['mode']['enum'] == ['requested']
        stream = types.SimpleNamespace(platform='qq', account_id='test-bot', scope='group',
            group_id='test-group', group_name='test-group', user_id='', context=None, session_id='expression-test')
        chat = types.SimpleNamespace(get_session_by_session_id=lambda sid: stream if sid == stream.session_id else None)
        with patch.dict(sys.modules, {heart.__name__:heart}), \
             patch.object(send, '_chat_manager', chat), \
             patch.object(send, 'get_bot_account', return_value='test-bot'), \
             patch.object(send, '_invoke_send_hook', hook), \
             patch.object(send, 'get_platform_io_manager', return_value=platform), \
             patch.object(send.MessageUtils, 'store_message_to_db_async', store), \
             patch.object(send, '_dispatch_adapter_callbacks', AsyncMock()), \
             patch.object(send, '_notify_memory_automation_on_message_sent', AsyncMock()):
            for handler, emotion in [(plugin.happy,'开心'), (plugin.serious,'无奈认真'), (plugin.unexpected,'意料之外')]:
                result = await handler(mode='requested', stream_id=stream.session_id)
                assert result['sent'], result
                segment = platform.captured[-1]
                assert len(segment)==1 and segment[0].type=='emoji'
                assert base64.b64decode(segment[0].data) == (Path('/plugin/assets')/module.EXPRESSIONS[emotion][0]).read_bytes()
                history = runtime._chat_history[-1]
                assert emotion in build_visible_text_from_sequence(history.raw_message)
                assert history.message_id == stored[-1].message_id == result['message_id']
            platform.succeeds = False
            assert not (await plugin.happy(mode='requested', stream_id=stream.session_id))['sent']
            assert len(stored)==len(runtime._chat_history)==3

            voice_module = load('voice', '/tts/plugin.py')
            voice = voice_module.create_plugin()
            voice._audio = AsyncMock(return_value=b'fake audio')
            class Supervisor:
                async def invoke_plugin(self, method, pid, name, args, **kwargs):
                    handler = plugin.reply_expression if pid == 'rvosy.navi-expressions' else voice.reply_voice
                    return types.SimpleNamespace(error=None, payload={'success': True, 'result': await handler(**args)})
            supervisor = Supervisor()
            available = {}
            for instance, pid in [(plugin, 'rvosy.navi-expressions'), (voice, 'rvosy.sakura-tts')]:
                for component in instance.get_components():
                    if component['type'] == 'REPLY_EXTENSION':
                        entry = ReplyExtensionEntry(component['name'], component['type'], pid, component['metadata'])
                        available[entry.full_name] = extensions.ReplyExtensionTarget(supervisor, entry)
            context = ToolAvailabilityContext(session_id=stream.session_id, stream_id=stream.session_id, platform='qq')
            with patch.object(extensions, 'get_reply_extensions', return_value=available):
                schema = extensions.build_reply_extensions_schema(context)
                choice = schema['properties']['rvosy.navi-expressions.expression']
                assert choice['required'] == ['emotion']
                assert set(choice['properties']['emotion']['enum']) == set(module.EXPRESSIONS)
                execution = extensions.ReplyExtensionExecution(context, 'target', 'call')
                # Reverse order deliberately: the Host must run voice before expression.
                await execution.prepare({'rvosy.navi-expressions.expression': {'emotion': '开心', 'mode': 'requested'},
                                         'rvosy.sakura-tts.voice': {'mode': 'requested'}})
                text = json.dumps({'text': '谢谢。', 'voice_ja': 'ありがとう'}, ensure_ascii=False)
                transformed = await execution.transform(text, [{'segments': [{'type':'text','data':text}], 'quote_previous':True}])
                assert [m['segments'][0]['type'] for m in transformed] == ['text','voice','emoji']
                assert transformed[0]['segments'][0]['data'] == '谢谢。'
                assert transformed[0]['quote_previous'] is True
                platform.succeeds = True
                for item in transformed:
                    sent = await host._cap_send_hybrid('rvosy.navi-expressions', 'send.hybrid', {
                        'segments':item['segments'], 'stream_id':stream.session_id, 'show_log':False,
                        'sync_to_maisaka_history':True, 'maisaka_source_kind':'guided_reply'})
                    assert sent['success']
                assert [x[0].type for x in platform.captured[-3:]] == ['text','voice','emoji']
                assert 'last_sent' in plugin._state[stream.session_id]
                no_attachment = extensions.ReplyExtensionExecution(context, 'target', 'call2')
                await no_attachment.prepare({})
                ordinary = [{'segments':[{'type':'text','data':'普通回复'}], 'quote_previous':False}]
                assert await no_attachment.transform('普通回复', ordinary) == ordinary
        await plugin.on_unload()
    report={'status':'passed','three_original_images':True,'real_host_schema':True,'success_only_history':True,
            'qq_sticker_sub_type':1,'reply_extension_voice_and_sticker_order':True,
            'platform_messages_sent':0,'production_database_writes':0}
    Path('/eval/reply-host-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


asyncio.run(main())
