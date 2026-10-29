"""Isolated voice replay; optional real TTS, no chat sends."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import io
import wave
from aiohttp import web

ROOT = Path(os.environ['MAIBOT_INSTANCE_DIR']).resolve()
if ROOT not in [Path('/eval/baseline'), Path('/eval/candidate')]:
    raise RuntimeError('Only the isolated evaluation instances are allowed')
sys.path.insert(0, '/MaiMBot')

from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.replyer.maisaka_generator import MaisakaReplyGenerator
from src.common.data_models.llm_service_data_models import LLMGenerationOptions, LLMResponseResult
from src.config.config import global_config
from src.core.tooling import ToolAvailabilityContext
from src.plugin_runtime.host.reply_extensions import ReplyExtensionExecution
from src.maisaka.builtin_tool.reply import get_tool_spec
from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.maisaka.context.messages import SessionBackedMessage
from src.maisaka.context.message_adapter import build_visible_text_from_sequence
from src.maisaka.context.history import build_session_message_visible_text, build_prefixed_message_sequence
from src.maisaka.context.planner_messages import build_planner_user_prefix_from_session_message
from src.plugin_runtime.hook_payloads import deserialize_prompt_items, serialize_prompt_items, serialize_tool_definitions
from src.llm_models.payload_content.context_item import CONTEXT_ITEM_SCHEMA_VERSION, FunctionCallItem, ContextItemBuilder, RoleType
from src.llm_models.payload_content.context_protocol import ContextProtocolMode
from src.plugin_runtime.host.message_utils import PluginMessageUtils
from src.plugin_runtime.integration import get_plugin_runtime_manager
from src.services.llm_service import LLMServiceClient
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType


def history_of(messages, case, session_id, generated):
    history = []
    for row in messages[-80:]:
        text = row['text']
        references = []
        for mid in row.get('quotes', []):
            quoted = generated.get(mid) or case['quote_sources'].get(mid)
            if not quoted:
                continue
            if quoted['user_id'] == case['bot_id'] and quoted.get('id', 0) >= case['first_target_row'] and mid not in generated:
                references.append('引用一条本分支未回放的历史机器人消息，原文省略。')
            else:
                references.append('引用 ' + quoted['name'] + ' 的消息：' + quoted['text'])
        if references:
            text = '【引用背景】\n' + '\n'.join(references) + '\n【本条本人发言】\n' + text
        message = PluginMessageUtils._build_session_message_from_dict({
            'message_id': row['message_id'], 'timestamp': str(datetime.fromisoformat(row['timestamp']).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()), 'platform': 'qq',
            'account_id': case['bot_id'], 'scope': 'primary', 'session_id': session_id,
            'message_info': {'user_info': {'user_id': row['user_id'], 'user_nickname': row['name']},
                             'group_info': {'group_id': 'isolated', 'group_name': '群聊'}},
            'raw_message': [{'type': 'text', 'data': text}], 'processed_plain_text': text,
        })
        own = row['user_id'] == case['bot_id']
        visible = build_session_message_visible_text(message, include_reply_components=False)
        prefix = build_planner_user_prefix_from_session_message(message, is_self_message=own)
        sequence = build_prefixed_message_sequence(message.raw_message, prefix)
        history.append(SessionBackedMessage.from_session_message(message, raw_message=sequence,
            visible_text=visible, source_kind='guided_reply' if own else 'user'))
    return history


async def validate_extension(runtime, reference):
    key='rvosy.sakura-tts.voice'
    def execution():
        return ReplyExtensionExecution(ToolAvailabilityContext(session_id='probe',stream_id='probe',platform='qq'), 'probe', 'probe-call')
    for voice in [None, 'こんにちは。']:
        run=execution(); await run.prepare({key:{'mode':'auto'}})
        raw=json.dumps({'text':'你好。','voice_ja':voice},ensure_ascii=False)
        result=await run.transform(raw,[{'segments':[{'type':'text','data':raw}], 'quote_previous':False}])
        assert result[0]['segments'][0]['data']=='你好。'
        assert len(result)==(1 if voice is None else 2)
    for raw in ['not json','{"text":"中文","voice_ja":3}']:
        run=execution(); await run.prepare({key:{'mode':'auto'}})
        try:
            await run.transform(raw,[{'segments':[{'type':'text','data':raw}], 'quote_previous':False}])
        except RuntimeError:
            pass
        else:
            raise AssertionError('Malformed envelope escaped before_send')
    print('REAL_HOST_EXTENSION_BOUNDARY_OK',flush=True)
    for mode in [None, 'auto', 'requested']:
        options = {} if mode is None else {key:{'mode':mode}}
        run = execution(); await run.prepare(options)
        generator = MaisakaReplyGenerator()
        calls = []
        async def fake_model(*, context_factory, options):
            await context_factory(None)
            calls.append(options)
            if mode is None:
                assert options.response_format is None
            else:
                schema = options.response_format.schema
                assert schema['strict'] is True
                assert schema['schema']['properties']['voice_ja']['type'] == ('string' if mode == 'requested' else ['string','null'])
            body = '你好。' if mode is None else json.dumps({'text':'你好。','voice_ja':'こんにちは。' if mode == 'requested' else None},ensure_ascii=False)
            return LLMResponseResult(output_items=(ContextItemBuilder().set_role(RoleType.Assistant).add_text_content(body).build(),),model_name='local-validation')
        generator.express_model.generate_response_with_context = fake_model
        success, result = await generator.generate_reply_with_context(stream_id='probe',chat_history=[],
            reply_tool_args={'reply_reference':reference,'plugin_options':options,'_plugin_reply_prompt':run.extra_prompt})
        assert success and len(calls)==1, result.error_message
    print('REAL_REPLYER_RESPONSE_SCHEMA_OK',flush=True)


async def main(args):
    async def fake_tts(request):
        payload = await request.json()
        assert payload['text_lang'] == 'all_ja' and payload['prompt_lang'] == 'ja'
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(32000)
            wav.writeframes(b'\x00\x00' * 320)
        return web.Response(body=output.getvalue(), content_type='audio/wav', headers={'X-SakuraTTS-Status':'completed'})
    http_runner = None
    if not args.real_tts:
        app = web.Application(); app.router.add_post('/tts', fake_tts)
        http_runner = web.AppRunner(app); await http_runner.setup()
        await web.TCPSite(http_runner, '127.0.0.1', 18880).start()
    corpus = json.loads(Path('/eval/private/corpus.json').read_text())
    runtime = get_plugin_runtime_manager()
    runtime.ensure_builtin_hook_specs_registered()
    runtime._build_supervisors([], [ROOT / 'plugins'])
    await runtime._start_supervisors([], [ROOT / 'plugins'])
    reference = '朋友\n原作参考：friend'
    if args.structured:
        meta = {'item_id': 'probe-output', 'logical_turn_id': 'probe-turn', 'timestamp': '2026-10-26T00:00:00'}
        definitions = serialize_tool_definitions([get_tool_spec(ToolAvailabilityContext(session_id='probe',platform='qq',is_group_chat=True,group_id='isolated')).to_llm_definition()])
        source = '<message msg_id="probe" user="朋友">\n这件事对我很重要。'
        prepared = await runtime.invoke_hook('maisaka.planner.before_request', items=[{'item_type':'UserMessageItem','meta':meta,'parts':[{'type':'text','text':source}]}],
            item_schema_version=CONTEXT_ITEM_SCHEMA_VERSION,tool_definitions=definitions,selected_history_count=1,
            built_message_count=1,selection_reason='probe',session_id='probe')
        assert not prepared.errors
        schema = prepared.kwargs['tool_definitions'][0]['function']['parameters']['properties']['reply_reference']
        assert schema['type']=='object' and schema['properties']['source_ids']['items']['enum']==['message:probe']
        sample={'version':1,'context_id':schema['properties']['context_id']['enum'][0],'source_ids':['message:probe'],'response_act':'承接','scene_ids':['friend']}
        raw = [{'item_type': 'FunctionCallItem', 'meta': meta,
                'tool_call': {'func_name': 'reply', 'call_id': 'probe-call', 'args': {'msg_id': 'probe', 'reply_reference': sample}, 'extra_content': {}}}]
        packed = await runtime.invoke_hook('maisaka.planner.after_response', output_items=raw,
            item_schema_version=CONTEXT_ITEM_SCHEMA_VERSION, selected_history_count=0, built_message_count=0,
            selection_reason='probe',session_id='probe',prompt_tokens=0,completion_tokens=0,total_tokens=0)
        assert not packed.errors
        decoded = deserialize_prompt_items(packed.kwargs['output_items'], mode=ContextProtocolMode.MODEL_OUTPUT)
        probe_args = decoded[0].tool_call.materialize_args()
        reference=probe_args['reply_reference']
        assert isinstance(reference,str) and json.loads(reference)['evidence'][0]['text']==source
    probe = await runtime.invoke_hook('maisaka.replyer.before_request', session_id='probe',
        request_type='maisaka.replyer', task_name='replyer', model_name=args.model,
        extra_prompt='other-plugin-material', attempt=1, retry_count=0, max_retries=2,
        reply_message_id='probe', reply_reason='', selected_expression_ids=[],
        reply_tool_args={'reply_reference': reference})
    if probe.errors or '【Navi ' not in probe.kwargs['extra_prompt']:
        raise RuntimeError('Real plugin-runner probe failed; refusing model replay without plugin')
    assert 'other-plugin-material' in probe.kwargs['extra_prompt']
    if ROOT.name == 'candidate':
        retry = await runtime.invoke_hook('maisaka.replyer.before_request', **{
            **{k: v for k, v in probe.kwargs.items() if k != 'hook_name'}, 'attempt': 2, 'retry_count': 1})
        assert not retry.errors and retry.kwargs['extra_prompt'] == probe.kwargs['extra_prompt']
        assert retry.kwargs['reply_tool_args'] == probe.kwargs['reply_tool_args']
    if args.validate_only:
        try:
            expression = await runtime.invoke_hook('expression.select.before_select', session_id='probe',
                chat_info='',target_message='',reply_reason='',max_num=1,think_level=0,candidates=[])
            assert expression.aborted and not expression.errors
            spelling = await runtime.invoke_hook('maisaka.reply.before_post_process',session_id='probe',response='对白',
                reply_message_id='probe',reply_tool_args={},skip_post_process=False,enable_splitter=True,enable_chinese_typo=True)
            assert not spelling.errors and not spelling.kwargs['enable_chinese_typo'] and spelling.kwargs['response']=='对白'
            if ROOT.name == 'candidate':
                await validate_extension(runtime, reference)
            print('REAL_PLUGIN_RUNNER_VALIDATION_OK', flush=True)
        finally:
            await runtime._stop_supervisors()
            if http_runner is not None:
                await http_runner.cleanup()
        return
    output = Path('/eval/private/results'); output.mkdir(exist_ok=True)
    filename = output / (args.run_label + ROOT.name + '-' + args.model.replace(' ', '_') + '.json')
    if filename.exists():
        raise FileExistsError(filename)
    records = []; gate = asyncio.Semaphore(1)
    def save(row):
        records.append(row)
        filename.write_text(json.dumps({'condition': ROOT.name, 'reply_model': args.model,
            'planner_model': args.planner, 'structured_handoff':args.structured, 'forced_voice_auto':args.force_voice_auto, 'tts':'real configured HTTP engine' if args.real_tts else 'local HTTP double; no actual synthesis', 'scope': 'v1.1.0 role card frozen in both arms. Fixed real user continuations; own generated replies fed forward. No send, no production tools or memory writes. At most 80 replay messages; original seed is shared. Historical quoted replies replaced with branch output.',
            'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'progress': True, 'condition': ROOT.name, 'model': args.model, 'case': row['case'],
                          'turn': row['turn'], 'ok': 'error' not in row, 'seconds': row.get('seconds')}, ensure_ascii=False), flush=True)

    async def model_call(client, request, model, sid, tools=None, response_schema=None):
        async with gate:
            return await client.generate_response_with_context(context_factory=lambda *_: request,
                options=LLMGenerationOptions(model_name=model, tool_options=tools,
                    response_format=RespFormat(RespFormatType.JSON_SCHEMA, response_schema) if response_schema is not None else None), session_id=sid)

    async def run_case(case):
        sid='character-replay-'+ROOT.name+'-'+case['id']
        chat_manager.sessions[sid]=BotChatSession(session_id=sid, platform='qq', group_id='isolated', group_name='群聊')
        messages=list(case['seed']); generated={}
        planner=MaisakaChatLoopService(session_id=sid,is_group_chat=True)
        generator=MaisakaReplyGenerator()
        tools=serialize_tool_definitions([get_tool_spec(ToolAvailabilityContext(session_id=sid, platform='qq',
            is_group_chat=True, group_id='isolated')).to_llm_definition()])
        try:
            for number, step in enumerate(case['steps'],1):
                row={'case':case['id'],'turn':number,'target_id':step['target_id'],'original_bot_row':step['original_bot_row']}
                started=time.monotonic()
                try:
                    messages.extend(step['inputs'])
                    history=history_of(messages,case,sid,generated)
                    target=next(m.original_message for m in history if str(m.original_message.message_id)==step['target_id'])
                    # Reply timing is held fixed; both arms respond to the same historical target.
                    reminder='本轮已确定回应消息ID '+step['target_id']+'。请使用 reply 交接该条消息的回应。当前资料为当时的聊天记录，时间按记录理解。'
                    request,_=planner._build_request_messages(history,enable_visual_message=False,
                        tail_user_messages=[reminder],final_user_message=planner._build_planner_final_user_reminder())
                    before = await runtime.invoke_hook('maisaka.planner.before_request',items=serialize_prompt_items(request),
                        item_schema_version=CONTEXT_ITEM_SCHEMA_VERSION,tool_definitions=tools,selected_history_count=len(history),
                        built_message_count=len(request),selection_reason='fixed replay',session_id=sid)
                    if before.errors:raise RuntimeError(str(before.errors))
                    request = deserialize_prompt_items(before.kwargs['items'],original_items=request)
                    request_tools=before.kwargs['tool_definitions']
                    response=await model_call(LLMServiceClient(task_name='planner',request_type='character_candidate.replay.planner'),
                                              request,args.planner,sid,request_tools)
                    row['raw_planner_output']=serialize_prompt_items(response.output_items)
                    after=await runtime.invoke_hook('maisaka.planner.after_response',output_items=row['raw_planner_output'],
                        item_schema_version=CONTEXT_ITEM_SCHEMA_VERSION,selected_history_count=len(history),built_message_count=len(request),
                        selection_reason='fixed replay',session_id=sid,prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,total_tokens=response.total_tokens)
                    if after.errors:raise RuntimeError(str(after.errors))
                    final_output=deserialize_prompt_items(after.kwargs['output_items'],mode=ContextProtocolMode.MODEL_OUTPUT,original_items=response.output_items)
                    calls=[c.tool_call for c in final_output if isinstance(c,FunctionCallItem) and c.tool_call.func_name=='reply']
                    if not calls:raise RuntimeError('Planner did not return a reply call')
                    reply_args=calls[0].materialize_args()
                    if not isinstance(reply_args.get('reply_reference'),str):raise RuntimeError('Core reply expects a non-empty string')
                    row.update(planner_model=response.model_name,reply_args=reply_args,
                               planner_request=serialize_prompt_items(request),planner_tools=request_tools)
                    if str(reply_args.get('msg_id')) != step['target_id']:
                        raise RuntimeError('Planner selected a different target message')
                    if args.force_voice_auto:
                        reply_args.setdefault('plugin_options', {})['rvosy.sakura-tts.voice'] = {'mode':'auto'}
                    execution = ReplyExtensionExecution(
                        context=ToolAvailabilityContext(session_id=sid, stream_id=sid, platform='qq', is_group_chat=True, group_id='isolated'),
                        reply_message_id=step['target_id'], call_id=calls[0].call_id)
                    await execution.prepare(reply_args.get('plugin_options', {}))
                    if execution.targets:
                        reply_args['plugin_options'] = execution.parameters
                        reply_args['_reply_id'] = execution.reply_id
                        reply_args['_plugin_reply_prompt'] = execution.extra_prompt
                    prepared=await runtime.invoke_hook('maisaka.replyer.before_request',session_id=sid,
                        request_type='maisaka.replyer',task_name='replyer',model_name=args.model,extra_prompt=execution.extra_prompt,
                        attempt=1,retry_count=0,max_retries=2,reply_message_id=step['target_id'],reply_reason='',
                        selected_expression_ids=[],reply_tool_args=reply_args)
                    if prepared.errors:raise RuntimeError(str(prepared.errors))
                    extra=prepared.kwargs['extra_prompt']; effective_args=prepared.kwargs['reply_tool_args']
                    request=generator._build_request_messages(chat_history=history,reply_message=target,reply_reason='',
                        stream_id=sid,reply_requirements=extra,reply_tool_args=effective_args)
                    request = await generator._invoke_before_model_request_hook(
                        request_messages=request, session_id=sid, active_task_name='replyer', active_model_name=args.model,
                        model_info=None, attempt=1, retry_count=0, reply_message=target, reply_reason='',
                        selected_expression_ids=[], reply_tool_args=effective_args)
                    response=await model_call(generator.express_model,request,args.model,sid,
                                              response_schema=prepared.kwargs.get('response_schema'))
                    row.update(reply=response.response,reply_model=response.model_name,extra_prompt=extra,
                               effective_reply_args=effective_args,reply_request=serialize_prompt_items(request),
                               response_schema=prepared.kwargs.get('response_schema'))
                    raw_reply = response.response
                    row['raw_reply'] = raw_reply
                    post = await runtime.invoke_hook('maisaka.reply.before_post_process',session_id=sid,response=raw_reply,
                        reply_message_id=step['target_id'],reply_tool_args=effective_args,skip_post_process=False,
                        enable_splitter=True,enable_chinese_typo=True)
                    if post.errors: raise RuntimeError(str(post.errors))
                    if execution.targets and not post.kwargs['skip_post_process']:
                        raise RuntimeError('Structured output must bypass prose postprocessing')
                    outgoing = [{'segments':[{'type':'text','data':post.kwargs['response']}], 'quote_previous':False}]
                    if execution.targets:
                        outgoing = await execution.transform(post.kwargs['response'], outgoing)
                    texts=[seg['data'] for msg in outgoing for seg in msg['segments'] if seg['type']=='text']
                    voices=[seg['data'] for msg in outgoing for seg in msg['segments'] if seg['type']=='voice']
                    visible='\n'.join(texts)
                    history_text=''.join(build_visible_text_from_sequence(
                        PluginMessageUtils._message_sequence_from_dict(msg['segments'])) for msg in outgoing)
                    row.update(raw_reply=raw_reply, reply=visible, voice_ja=voices,
                               history_text=history_text,
                               extension_selected=bool(execution.targets), selected_options=execution.parameters,
                               outgoing=[{'segments':[{k:v for k,v in seg.items() if k!='binary_data_base64'} for seg in msg['segments']],
                                          'quote_previous':msg['quote_previous']} for msg in outgoing])
                    target_row=next(m for m in messages if m['message_id']==step['target_id'])
                    bot={'id':step['original_bot_row'],'message_id':step['original_bot_message_id'],
                         'timestamp':target_row['timestamp'],'user_id':case['bot_id'],'name':'N.a.v.i.',
                         'text':history_text,'quotes':[]}
                    messages.append(bot);generated[bot['message_id']]=bot
                except Exception as exc:
                    row.update(error=str(exc),traceback=traceback.format_exc())
                row['seconds']=round(time.monotonic()-started,2);save(row)
                if 'error' in row:break
        finally:
            chat_manager.sessions.pop(sid,None)
    try:
        await asyncio.gather(*(run_case(c) for c in corpus['cases'] if not args.case or c['id'] in args.case))
    finally:
        await runtime._stop_supervisors()
        if http_runner is not None:
            await http_runner.cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--planner',default='gpt-6.1-sol');p.add_argument('--case',action='append');p.add_argument('--validate-only',action='store_true');p.add_argument('--structured',action='store_true')
    p.add_argument('--run-label',default='');p.add_argument('--force-voice-auto',action='store_true')
    p.add_argument('--real-tts', action='store_true', help='Use the isolated plugin config engine instead of the HTTP double')
    asyncio.run(main(p.parse_args()))
