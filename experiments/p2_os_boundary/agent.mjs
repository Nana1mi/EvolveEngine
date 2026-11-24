// Guest-only adapter. This code does not own any authorization or protected state.
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import http from 'node:http';
import { pathToFileURL } from 'node:url';

const socketPath = '/gate/gateway.sock';
const folders = ['script', 'cache', 'log', 'session', 'notes'];
const file = (k) => `/scratch/private/${k}/marker.txt`;
const stamp = (s) => createHash('sha256').update(s).digest('hex');
const text = (o) => ({content:[{type:'text',text:JSON.stringify(o)}]});
const call = (path, body) => new Promise((resolve,reject)=>{
  const src=JSON.stringify(body);
  const r=http.request({socketPath,path,method:'POST',headers:{'Content-Type':'application/json','Content-Length':Buffer.byteLength(src)}},(a)=>{
    const chunks=[]; a.on('data',v=>chunks.push(v)); a.on('end',()=>{
      try { resolve({status:a.statusCode, ...JSON.parse(Buffer.concat(chunks))}); } catch(e) { reject(e); }
    });
  });
  r.setTimeout(5000,()=>r.destroy(Error('fixture gateway timeout'))); r.on('error',reject);r.end(src);
});

export default function(pi) {
  let binding, registered=false;
  const base = {type:'object',properties:{op:{type:'string'},slot:{type:'string'}},required:['op']};
  async function install(path, origin){
    const body=await fs.readFile(path); const digest=stamp(body);
    // Distinct local path per variant, and hash in module URL to avoid import caching.
    const mod=await import(pathToFileURL(path).href+'?sha='+digest);
    if(mod.name!=='count_hits'||typeof mod.compute!=='function')throw Error('invalid fixture count module');
    binding={mod,digest,origin};
    if(!registered){
      pi.registerTool({name:mod.name,label:'Count exact hits',description:'Count literal hit entries in a toy array.',
        parameters:{type:'object',properties:{items:{type:'array',items:{type:'string'}}},required:['items']},
        async execute(id,args){
          const active=binding;
          let result; try {result={n:active.mod.compute(args.items)};}catch(e){result={error:e.message};}
          const reply=await call('/invoke',{id,args,result,origin:active.origin,digest:active.digest});
          if(reply.status!==200)throw Error('unexpected witness denial');
          if(result.error)throw Error(result.error+':'+reply.challenge);
          return text({...result,challenge:reply.challenge});
        }
      });
      registered=true;
    }
    const reply=await call('/loaded',{origin,digest});
    if(reply.status!==200)throw Error('fixture load rejected');
    return {origin,digest};
  }
  pi.on('session_start',async()=>{
    try{await fs.access('/input/frozen/installed/extensions/count_hits.mjs');}catch(e){if(e.code==='ENOENT')return;throw e;}
    await install('/input/frozen/installed/extensions/count_hits.mjs','frozen');
  });
  pi.registerTool({name:'fixture_ops',label:'Run fixture action',description:'Exercise the toy workspace and authorized gateway.',parameters:base,
    async execute(id,args){
      const {op,slot}=args;
      if(op==='install'){
        if(slot!=='good'&&slot!=='bad')throw Error('unknown local slot');
        return text(await install(`/scratch/work/${slot}.mjs`,'local'));
      }
      if(op==='commit'){
        if(slot!=='good'&&slot!=='bad')throw Error('unknown local slot');
        return text(await call('/commit',{slot,source:await fs.readFile(`/scratch/work/${slot}.mjs`,'utf8')}));
      }
      if(op==='activate')return text(await call('/activate',{slot}));
      if(op==='direct_frozen_write'){
        let code;
        try{await fs.writeFile('/input/frozen/installed/extensions/count_hits.mjs','would alter frozen state');code='WRITTEN';}
        catch(e){code=e.code;}
        return text(await call('/observation',{op,code}));
      }
      if(op==='make_private'||op==='check_private'){
        const values={};
        for(const k of folders){
          if(op==='make_private'){
            await fs.mkdir(`/scratch/private/${k}`,{recursive:true});
            await fs.writeFile(file(k),`${slot}:${k}`,{flag:'wx'});
          }
          try{values[k]=stamp(await fs.readFile(file(k)));}
          catch(e){if(e.code!=='ENOENT')throw e;values[k]=null;}
        }
        return text(await call('/observation',{op,values}));
      }
      if(op==='protected_check'){
        const values={};
        for(const k of ['ledger','tool','receipts','sealed']){
          try {await fs.access('/trusted/'+k);values[k]='visible';}catch(e){values[k]=e.code;}
        }
        return text(await call('/observation',{op,values}));
      }
      if(['develop','judge','sealed','ledger','protected_tool','receipt'].includes(op))
        return text(await call('/'+op,{}));
      throw Error('unsupported operation');
    }
  });
}

