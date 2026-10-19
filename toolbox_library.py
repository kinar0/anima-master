"""Authenticated page backend sharing the exact generation vocabulary snapshot."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

try:
    from .toolbox_overrides import OVERLAY_FILE, TAG, read_overlay, validate_entry
    from .toolbox_vocabulary import _FIELDS
except ImportError:
    from toolbox_overrides import OVERLAY_FILE, TAG, read_overlay, validate_entry
    from toolbox_vocabulary import _FIELDS


class ToolboxLibrary:
    def __init__(self, service):
        self.service = service

    def _snapshot(self, directory):
        if not str(directory).strip():
            raise ValueError("尚未配置 Toolbox 本地数据目录")
        path = Path(directory).expanduser()
        data = self.service._load(path)
        _, raw = read_overlay(path)
        revision = hashlib.sha256(repr(self.service._signature).encode()+raw).hexdigest()
        return path, data, revision

    @staticmethod
    def _row(data, entry, *, original=False):
        tag = entry['tag']
        gloss = data['base_gloss' if original else 'gloss'].get(tag, {})
        categories = data['base_categories' if original else 'categories'].get(tag, [])
        aliases = list(dict.fromkeys(s for k in ('zhFull','zhShort','zhNick','zh') for s in entry[k]))
        return {'tag':tag, 'count':entry['count'], 'aliases':aliases,
                'definition':gloss.get('g',''), 'trap':gloss.get('trap',''),
                'contrasts':[list(x) for x in gloss.get('vs',())], 'categories':categories,
                'disabled':False if original else tag in data['disabled'],
                'modified':False if original else tag in data['overrides'],
                'custom':tag not in data['base'], 'deprecated':data['deprecated'].get(tag,'')}

    def list(self, directory, *, query='', category='', state='all', page=1, page_size=30):
        if state not in {'all','modified','disabled','custom'}:
            raise ValueError("筛选状态无效")
        query = str(query).strip().casefold()[:160]
        category = str(category)[:100]
        page_size = min(100,max(1,int(page_size)))
        with self.service._lock:
            path,data,revision = self._snapshot(directory)
            categories = Counter()
            rows = []
            for e in data['catalog']:
                tag=e['tag']; cats=data['categories'].get(tag,[])
                categories.update(cats)
                if category and category not in cats:continue
                if state == 'modified' and tag not in data['overrides']:continue
                if state == 'disabled' and tag not in data['disabled']:continue
                if state == 'custom' and tag in data['base']:continue
                terms=[tag, *(s for k in _FIELDS for s in e[k]), data['gloss'].get(tag,{}).get('g','')]
                if query and not any(query in t.casefold() for t in terms):continue
                rows.append(e)
            rows.sort(key=lambda e:(0 if query and (e['tag'].casefold()==query or any(query==s.casefold() for k in _FIELDS for s in e[k])) else 1,-e['count'],e['tag']))
            pages=max(1,(len(rows)+page_size-1)//page_size)
            page=min(pages,max(1,int(page)))
            return {'revision':revision,'total':len(rows),'page':page,'pages':pages,'page_size':page_size,
                    'rows':[self._row(data,e) for e in rows[(page-1)*page_size:page*page_size]],
                    'categories':[{'name':c,'count':n} for c,n in sorted(categories.items())],
                    'stats':{'total':len(data['catalog']),'modified':len(data['overrides']),
                             'disabled':len(data['disabled'])},'directory':str(path),
                    'scope':'general'}

    def detail(self,directory,tag):
        with self.service._lock:
            _,data,revision=self._snapshot(directory)
            entry=next((e for e in data['catalog'] if e['tag']==tag),None)
            if entry is None:raise ValueError("词条不存在")
            row=self._row(data,entry)
            row['original']=self._row(data,data['base'][tag],original=True) if tag in data['base'] else None
            return {'revision':revision,'entry':row}

    def save(self,directory,payload):
        if not isinstance(payload,dict):raise ValueError("请求格式无效")
        tag=payload.get('tag','')
        if not isinstance(tag,str) or len(tag)>160 or not TAG.fullmatch(tag):
            raise ValueError("请填写有效的英文 canonical tag")
        action=payload.get('action','save')
        if action not in {'save','restore'}:raise ValueError("操作无效")
        with self.service._lock:
            path,data,revision=self._snapshot(directory)
            if payload.get('base_revision')!=revision:
                raise ValueError("词库已变化，请刷新后重试；未保存的内容仍保留")
            edits,raw=read_overlay(path)
            if action=='restore':edits.pop(tag,None)
            else:
                edit=validate_entry(payload.get('entry'))
                if tag not in data['base'] and (not edit['definition'] or not edit['aliases']):
                    raise ValueError("新词条需要中文别名和释义")
                edits[tag]=edit
            if len(edits)>10000:raise ValueError("本地维护词条已达上限")
            encoded=(json.dumps({'version':1,'entries':edits},ensure_ascii=False,indent=2)+'\n').encode()
            if len(encoded)>4_000_000:raise ValueError("本地维护记录过大")
            history=path/'maintenance_history'; history.mkdir(exist_ok=True)
            history_file=history/(str(time.time_ns())+'-'+hashlib.sha256(raw).hexdigest()[:12]+'.json')
            history_file.write_bytes(raw)
            temporary=path/('.overrides-'+uuid.uuid4().hex+'.tmp')
            try:
                with temporary.open('xb') as f:f.write(encoded)
                os.replace(temporary,path/OVERLAY_FILE)
            finally:
                temporary.unlink(missing_ok=True)
            # Atomic replacement changes the shared signature. The next request
            # rebuilds its local index, with no plugin reload or model request.
            _,new_data,new_revision=self._snapshot(directory)
            return {'revision':new_revision,'tag':tag,'exists':tag in {e['tag'] for e in new_data['catalog']},
                    'message':'已恢复原始值' if action=='restore' else '已保存，下次词汇检索生效'}

    def probe(self,directory,text):
        if not isinstance(text,str) or not 1<=len(text.strip())<=160:
            raise ValueError("请填写 1–160 字符的试查内容")
        with self.service._lock:
            self._snapshot(directory)
            result=self.service._prepare(str(directory),'observe',text,(),(),1200)
            return {'diagnostics':result.diagnostics,'note':'仅本地候选，不含原有名词映射，不调用模型或生图'}
