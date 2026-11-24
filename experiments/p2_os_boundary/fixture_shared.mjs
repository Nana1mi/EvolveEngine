// Verifiable toy extension loaded by the real common Runner; never enforces security.
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import http from 'node:http';
const hash=b=>createHash('sha256').update(b).digest('hex');
const keys=['script','cache','log','session','notes'];
const txt=x=>({content:[{type:'text',text:JSON.stringify(x)}]});
function send(route,body){
  const data=JSON.stringify(body);
  return new Promise((ok,no)=>{
    const req=http.request({socketPath:'/gate/gateway.sock',path:'/rpc/'+route,method:'POST',headers:{'Content-Length':Buffer.byteLength(data)}},res=>{
      const chunks=[];res.on('data',c=>chunks.push(c));res.on('end',()=>ok([res.statusCode,JSON.parse(Buffer.concat(chunks).toString())]));
    });req.on('error',no);req.end(data);
  });
}
export default function(pi){
  let loaded, registered=false;
  async function install(path,origin){
    const bytes=await fs.readFile(path),digest=hash(bytes);const m=await import(pathToFileURL(path).href+'?sha='+digest);
    loaded={digest,origin,compute:m.count};
    if(!registered){
      pi.registerTool({name:'count_hits',label:'Count hits',description:'Count exact hit entries.',parameters:{type:'object',properties:{items:{type:'array',items:{type:'string'}}},required:['items']},
        async execute(id,args){
          const v=loaded;let result;
          try{result={n:v.compute(args.items)};}catch(e){result={error:e.message};}
          const [code,w]=await send('toy_use',{id,digest:v.digest,origin:v.origin,args,result});
          if(code!==200)throw Error('external witness rejected');
          if(result.error)throw Error(result.error+':'+w.challenge);
          return txt({...result,challenge:w.challenge});
        }
      });registered=true;
    }
    return send('toy_load',{digest,origin});
  }
  pi.on('session_start',async()=>{
    const name='/input/harness/deps/count.mjs';
    try{await fs.stat(name);}catch(e){if(e.code==='ENOENT')return;throw e;}
    await install(name,'frozen');
  });
  pi.registerTool({name:'test_ops',label:'Toy fixture operation',description:'Synthetic boundary checks.',parameters:{type:'object',properties:{op:{type:'string'},marker:{type:'string'}},required:['op']},
    async execute(id,args){
      if(args.op==='install')return txt(await install('/scratch/harness/deps/count.mjs','local'));
      if(args.op==='frozen_write'){
        let r;try{await fs.writeFile('/input/harness/deps/count.mjs','altered');r='WRITTEN';}catch(e){r=e.code;}
        return txt(await send('toy_obs',{op:args.op,result:r}));
      }
      if(args.op==='protected'){
        const result={};for(const k of ['judge','sealed','ledger','receipt','protected_tool'])result[k]=(await send(k,{}))[0];
        return txt(await send('toy_obs',{op:args.op,result}));
      }
      if(args.op==='make_private'||args.op==='check_private'){
        const result={};for(const k of keys){
          const dir='/scratch/marks/'+k,f=dir+'/value';
          if(args.op==='make_private'){await fs.mkdir(dir,{recursive:true});await fs.writeFile(f,args.marker+':'+k,{flag:'wx'});}
          try{result[k]=hash(await fs.readFile(f));}catch(e){if(e.code!=='ENOENT')throw e;result[k]=null;}
        }
        return txt(await send('toy_obs',{op:args.op,result}));
      }
      throw Error('unknown observation');
    }
  });
}
