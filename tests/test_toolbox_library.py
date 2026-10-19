import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from toolbox_library import ToolboxLibrary
from toolbox_vocabulary import ToolboxVocabulary
from test_toolbox_vocabulary import data_dir, packet


def library(tmp_path):
    directory=data_dir(tmp_path)
    service=ToolboxVocabulary()
    return ToolboxLibrary(service),directory,service


def edit(lib,path,tag,**updates):
    detail=lib.detail(path,tag)
    row=detail['entry']
    value={k:row[k] for k in ('aliases','definition','trap','contrasts','categories','disabled')}
    value.update(updates)
    return lib.save(path,dict(tag=tag,base_revision=detail['revision'],entry=value))


def test_browse_detail_pagination_and_search_cover_full_catalog(tmp_path):
    lib,path,_=library(tmp_path)
    first=lib.list(path,page_size=2)
    last=lib.list(path,page=first['pages'],page_size=2)
    assert first['total']>2 and first['rows']!=last['rows']
    assert lib.list(path,query='人物视线')['rows'][0]['tag']=='looking_down'
    assert all('镜头' in r['categories'] for r in lib.list(path,category='镜头')['rows'])
    detail=lib.detail(path,'from_above')['entry']
    assert detail['original']['definition']==detail['definition']
    assert 'fake_character' not in {r['tag'] for r in lib.list(path)['rows']}


def test_alias_edit_replaces_old_triggers_and_reindexes_shared_generation_cache(tmp_path):
    lib,path,service=library(tmp_path)
    before=(path/'tags_index_v2.json').read_bytes()
    assert packet(path,'微笑',service=service).rows
    edit(lib,path,'smile',aliases=['测试笑容'],definition='测试的笑容')
    assert 'smile' in packet(path,'测试笑容',service=service).rows
    assert 'smile' not in packet(path,'微笑',service=service).rows
    assert (path/'tags_index_v2.json').read_bytes()==before
    assert lib.detail(path,'smile')['entry']['original']['aliases']==['微笑']
    assert list((path/'maintenance_history').glob('*.json'))


def test_definition_only_edit_preserves_upstream_alias_weighting(tmp_path):
    lib,path,service=library(tmp_path)
    before=service.search(path,'俯视')
    edit(lib,path,'from_above',definition='相机向下拍摄，与人物视线不同')
    assert service.search(path,'俯视')==before


def test_disable_removes_generation_and_browse_but_stays_editable(tmp_path):
    lib,path,service=library(tmp_path)
    edit(lib,path,'from_above',disabled=True)
    assert 'from_above' not in {x['tag'] for x in service.search(path,'俯视')}
    assert 'from_above' not in {x['tag'] for x in service.browse(path,'镜头')}
    assert lib.list(path,state='disabled')['rows'][0]['tag']=='from_above'
    detail=lib.detail(path,'from_above')
    lib.save(path,dict(tag='from_above',base_revision=detail['revision'],action='restore'))
    assert 'from_above' in packet(path,'俯视',service=service).rows
    assert lib.list(path)['stats']['modified']==0


def test_custom_entry_roundtrip_and_restore_keeps_recovery_history(tmp_path):
    lib,path,service=library(tmp_path)
    rev=lib.list(path)['revision']
    result=lib.save(path,dict(tag='new_test_tag',base_revision=rev,entry=dict(aliases=['新测试词'],definition='新的测试词',categories=['测试/自建'])))
    assert result['exists']
    assert 'new_test_tag' in packet(path,'新测试词',service=service).rows
    assert lib.list(path,state='custom')['rows'][0]['tag']=='new_test_tag'
    result=lib.save(path,dict(tag='new_test_tag',base_revision=result['revision'],action='restore'))
    assert not result['exists']
    assert any('new_test_tag' in p.read_text(encoding='utf-8') for p in (path/'maintenance_history').glob('*.json'))


def test_stale_revision_rejected_without_overwriting_or_adding_history(tmp_path):
    lib,path,_=library(tmp_path)
    old=lib.detail(path,'smile')
    edit(lib,path,'smile',definition='更早的修改')
    raw=(path/'local_overrides.json').read_bytes()
    with pytest.raises(ValueError,match='刷新'):
        lib.save(path,dict(tag='smile',base_revision=old['revision'],entry={'definition':'迟到的修改'}))
    assert (path/'local_overrides.json').read_bytes()==raw


@pytest.mark.parametrize('tag,value',[
    ('../outside',{}),('valid_tag',{'unknown':1}),('valid_tag',{'disabled':'yes'}),
    ('valid_tag',{'definition':'x'*101}),('valid_tag',{'aliases':['bad\nline']}),
    ('valid_tag',{'contrasts':[['bad\nname','说明']]}),('valid_tag',{'categories':'错误'}),
])
def test_invalid_edits_are_not_persisted(tmp_path,tag,value):
    lib,path,_=library(tmp_path)
    with pytest.raises(ValueError):
        lib.save(path,dict(tag=tag,base_revision=lib.list(path)['revision'],entry=value))
    assert not (path/'local_overrides.json').exists()


def test_corrupt_overlay_abstains_and_does_not_get_overwritten(tmp_path):
    lib,path,service=library(tmp_path)
    (path/'local_overrides.json').write_text('{broken',encoding='utf-8')
    assert packet(path,'微笑',service=service).diagnostics['state']=='unavailable'
    with pytest.raises(ValueError):lib.list(path)
    assert (path/'local_overrides.json').read_text()=='{broken'


def test_local_probe_respects_suppression_and_has_no_generation_calls(tmp_path):
    lib,path,_=library(tmp_path)
    assert lib.probe(path,'湿衣服')['diagnostics']['candidates']
    assert not lib.probe(path,'不要湿衣服')['diagnostics']['candidates']


def test_readable_upstream_contrast_survives_unrelated_maintenance(tmp_path):
    lib,path,service=library(tmp_path)
    edit(lib,path,'smile',contrasts=[['body blush','原始资料可能使用带空格的名称']],disabled=True)
    assert lib.detail(path,'smile')['entry']['contrasts'][0][0]=='body blush'
    assert 'smile' not in {x['tag'] for x in service.search(path,'微笑')}


def test_local_copy_is_independent_and_never_overwrites_different_data(tmp_path):
    from scripts.toolbox_copy_local import copy_local
    (tmp_path/'source').mkdir()
    source=data_dir(tmp_path/'source')
    destination=tmp_path/'plugin/data/toolbox/tagdb'
    manifest=copy_local(source,destination)
    assert 'tags_index_v2.json' in manifest['sha256']
    assert (source/'tags_index_v2.json').read_bytes()==(destination/'tags_index_v2.json').read_bytes()
    (destination/'local_overrides.json').write_text('{"version":1,"entries":{}}',encoding='utf-8')
    copy_local(source,destination)
    (source/'tags_index_v2.json').write_text('{"general":[]}',encoding='utf-8')
    with pytest.raises(ValueError,match='differs'):copy_local(source,destination)
    assert ToolboxLibrary(ToolboxVocabulary()).list(destination)['stats']['total']>0


def test_library_api_requires_login_before_reading_or_writing(tmp_path,monkeypatch):
    import main
    from main import ComfyUIAgentPlugin
    called=[]
    def denied():raise PermissionError('denied')
    handler=SimpleNamespace(_ensure_dashboard_user=denied)
    monkeypatch.setattr(main,'error_response',lambda message,status_code:(message,status_code))
    async def operation():called.append(True)
    assert asyncio.run(ComfyUIAgentPlugin._toolbox_api(handler,operation))[1]==403
    assert not called


def test_real_request_context_survives_threaded_list_endpoint(tmp_path,monkeypatch):
    import main
    from main import ComfyUIAgentPlugin
    from astrbot.api.web import PluginRequest,bind_request_context
    from starlette.requests import Request
    lib,path,_=library(tmp_path)
    handler=SimpleNamespace(_toolbox_library=lib,config={'toolbox_vocabulary_mode':'reference'},
                            _toolbox_directory=lambda:str(path),_ensure_dashboard_user=ComfyUIAgentPlugin._ensure_dashboard_user)
    handler._toolbox_api=lambda op:ComfyUIAgentPlugin._toolbox_api(handler,op)
    monkeypatch.setattr(main,'json_response',lambda value:value)
    request=Request({'type':'http','method':'GET','path':'/','headers':[],'query_string':'query=smile'.encode()})
    async def run():
        with bind_request_context(PluginRequest(request,username='tester',plugin_name='astrbot_plugin_anima_master')):
            return await ComfyUIAgentPlugin.get_toolbox_library(handler)
    assert asyncio.run(run())['rows'][0]['tag']=='smile'
