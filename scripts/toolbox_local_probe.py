"""Measure a user-owned Toolbox dataset using synthetic requests, offline only."""
from __future__ import annotations
import argparse
import asyncio
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from toolbox_vocabulary import ToolboxVocabulary


def peak_working_set() -> int | None:
    if sys.platform!='win32':
        return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
                  *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
    counters=Counters()
    counters.cb=ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    handle=ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):
        return counters.PeakWorkingSetSize
    return None


async def probe(directory: Path) -> dict:
    service=ToolboxVocabulary()
    report={'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (directory/name for name in ('tags_index_v2.json','tag_gloss.json','tag_browse.json','tag_deprecated.json')) if p.exists()},
        'synthetic_only':True,'model_calls':0,'network_requests':0,'results':[]}
    for text in ('俯视','从上往下看','牛仔构图','湿衣服','不要湿衣服','人物低头，微笑',
                 '甲说：“湿衣服”','左格甲穿白衬衫，右格甲穿黑外套，俯视','xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'):
        start=time.perf_counter()
        packet=await service.prepare(directory=str(directory),mode='observe',text=text)
        report['results'].append({'synthetic_request':text,'wall_ms':round((time.perf_counter()-start)*1000,2),**packet.diagnostics})
    report['peak_process_working_set_bytes']=peak_working_set()
    report['working_set_note']='Peak for standalone probe process, including interpreter and JSON parsing; not incremental AstrBot memory.'
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report=asyncio.run(probe(args.data))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(args.report),'requests':len(report['results']),
                      'first_ms':report['results'][0]['wall_ms'],'peak_bytes':report['peak_process_working_set_bytes']},ensure_ascii=False))
