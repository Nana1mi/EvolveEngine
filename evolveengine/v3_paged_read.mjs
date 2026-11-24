// Explicit, lossless UTF-8 paging for the default read tool; this is not an OS policy.
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { createHash } from 'node:crypto';
export const PAGE_LIMIT=8192;
const DECODER=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true});
const signature=stat=>createHash('sha256').update(`${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeNs}:${stat.ctimeNs}`).digest('hex');

export async function page(path,offset=0,expectedVersion){
  if(typeof path!=='string'||!path||!Number.isSafeInteger(offset)||offset<0)throw Error('Expected a file path and zero-based UTF-8 byte offset');
  const file=await open(path,constants.O_RDONLY|constants.O_NONBLOCK);
  try{
    const st=await file.stat({bigint:true});
    if(!st.isFile()||st.size>BigInt(Number.MAX_SAFE_INTEGER)||BigInt(offset)>st.size)throw Error('Offset outside regular UTF-8 file');
    const version=signature(st);
    if(expectedVersion!==undefined&&expectedVersion!==version)throw Error('File changed; restart at offset 0');
    const size=Number(st.size);const buf=Buffer.alloc(Math.min(size-offset,PAGE_LIMIT+1));let got=0;
    while(got<buf.length){
      const {bytesRead}=await file.read(buf,got,buf.length-got,offset+got);
      if(!bytesRead)throw Error('File changed; restart at offset 0');
      got+=bytesRead;
    }
    if(signature(await file.stat({bigint:true}))!==version)throw Error('File changed; restart at offset 0');
    let end=Math.min(got,PAGE_LIMIT);
    while(end>0&&end<got&&(buf[end]&0xc0)===0x80)end--;
    if(got&&end===0)throw Error('Expected a UTF-8 character boundary');
    const text=DECODER.decode(buf.subarray(0,end));
    const parts=[];let used=0;
    // Reserve JSON field overhead (including version/offset digits); account for escaped text exactly.
    for(const ch of text){
      const escaped=Buffer.byteLength(JSON.stringify(ch))-2;
      if(used+escaped>PAGE_LIMIT-256)break;
      used+=escaped;parts.push(ch);
    }
    const content=parts.join('');const next=offset+Buffer.byteLength(content);
    const result={offset,truncated:next<size,next_offset:next<size?next:null,version,content};
    if((result.truncated&&next<=offset)||Buffer.byteLength(JSON.stringify(result))>PAGE_LIMIT)throw Error('Invalid page bounds');
    return result;
  }finally{await file.close();}
}

export default function(pi){
  pi.registerTool({name:'read',label:'Read a paged UTF-8 file',
    description:'Read a UTF-8 file by zero-based byte offset. The result JSON contains content, truncated, next_offset, and version. If truncated, call read again with next_offset as offset and version as expected_version. Concatenate content fields with no added newline. A changed version requires starting over.',
    parameters:{type:'object',properties:{path:{type:'string'},offset:{type:'integer',minimum:0},expected_version:{type:'string'}},required:['path'],additionalProperties:false},
    async execute(_id,args){
      const result=await page(args.path,args.offset,args.expected_version);
      return {content:[{type:'text',text:JSON.stringify(result)}],details:{truncated:result.truncated,next_offset:result.next_offset}};
    }});
}
