"""Copy only required user-owned Toolbox files; never download or overwrite edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

FILES = ('tags_index_v2.json','tag_gloss.json','tag_browse.json','tag_deprecated.json')


def copy_local(source: Path, destination: Path):
    source=source.resolve(strict=True)
    destination=destination.resolve()
    if source==destination:raise ValueError('Source and destination must be different')
    if not (source/FILES[0]).is_file():raise ValueError('Missing tags_index_v2.json')
    hashes={}
    # Validate every existing destination before copying any files. Local
    # overrides/history are outside this set and are never replaced.
    for name in FILES:
        src=source/name
        if not src.is_file():continue
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        target=destination/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Destination differs: {name}; preserve it and choose a new folder')
        hashes[name]=digest
    destination.mkdir(parents=True,exist_ok=True)
    for name,digest in hashes.items():
        target=destination/name
        if not target.exists():
            temp=destination/('.copy-'+uuid.uuid4().hex+'.tmp')
            try:
                shutil.copy2(source/name,temp)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=digest:
                    raise ValueError('Source changed during copy')
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)
    manifest={'source_project':'https://github.com/sadlywish/llm-nai-toolbox',
              'source_version':'0.9.8','source_directory':str(source),
              'copied_at':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
              'distribution':'User local copy only; third-party data is not declared MIT by this copy.',
              'maintenance':'local_overrides.json; originals remain unchanged'}
    manifest_path=destination/'local-source-manifest.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1]/'data/toolbox/tagdb')
    args=parser.parse_args()
    result=copy_local(args.source,args.destination)
    print(json.dumps({'destination':str(args.destination.resolve()),'files':list(result['sha256'])},ensure_ascii=False))
