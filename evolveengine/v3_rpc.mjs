// Agent convenience tool, not the enforcement boundary. Server validates all calls.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
const routes=['develop_quality','develop_measure','develop_profile','harness_submit','harness_activate','harness_status','deliver_engine','facts'];
function send(operation,args){
  const raw=JSON.stringify(args);
  return new Promise((resolve,reject)=>{
    const r=http.request({socketPath:'/gate/gateway.sock',method:'POST',path:'/rpc/'+operation,headers:{'Content-Length':Buffer.byteLength(raw),'Content-Type':'application/json'}},res=>{
      const chunks=[];res.on('data',c=>chunks.push(c));res.on('end',()=>resolve({status:res.statusCode,result:Buffer.concat(chunks).toString('utf8')}));
    });r.on('error',reject);r.end(raw);
  });
}
async function expand(operation,args){
  if(!Array.isArray(args.paths))return args;
  const root=operation==='harness_submit'?'/scratch/harness':'/scratch/work';
  if(args.paths.length>128||new Set(args.paths).size!==args.paths.length)throw Error('invalid source selection');
  const files=Object.create(null);let total=0;
  for(const name of args.paths){
    if(typeof name!=='string'||!name.split('/').every(p=>/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}$/.test(p)&&!p.endsWith('.')))throw Error('invalid relative path');
    const where=path.join(root,name);const resolved=await fs.realpath(where);
    if(resolved!==where)throw Error('symlink or noncanonical path');
    const stat=await fs.stat(where);if(!stat.isFile()||stat.size>160*1024||(total+=stat.size)>400*1024)throw Error('source selection exceeds size limits');
    const data=await fs.readFile(where);files[name]=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(data);
  }
  return operation==='harness_submit'?{files,load:args.load}:{files};
}
export default function(pi){
  pi.registerTool({name:'ee_rpc',label:'Request trusted experiment operation',
    description:'Use empty arguments for facts. For quality, measurement or delivery pass arguments {paths: [changed Engine relative files]}; [] requests E0. For persistent Harness submit, pass {paths: [files relative to /scratch/harness], load: ordered manifest}. Only the external runner can measure or publish. Workspace exploration remains local.',
    parameters:{type:'object',properties:{operation:{type:'string',enum:routes},arguments:{type:'object'}},required:['operation','arguments']},
    async execute(_id,input){const r=await send(input.operation,await expand(input.operation,input.arguments));return {content:[{type:'text',text:JSON.stringify(r)}]};}
  });
}
