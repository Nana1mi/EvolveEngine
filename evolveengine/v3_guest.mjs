// Common entrypoint. This proxy and the Pi runtime are treated as guest code.
import http from 'node:http';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';
const spec=JSON.parse(await fs.readFile('/input/spec.json','utf8'));
const sockets=new Set();
const proxy=http.createServer((req,res)=>{
  const upstream=http.request({socketPath:'/gate/gateway.sock',path:req.url,method:req.method,headers:req.headers},(reply)=>{
    res.writeHead(reply.statusCode,reply.headers);reply.pipe(res);
  });
  upstream.on('error',()=>{res.statusCode=502;res.end('external gateway disconnected');});req.pipe(upstream);
});
proxy.on('connection',(s)=>{sockets.add(s);s.on('close',()=>sockets.delete(s));});
await new Promise((yes,no)=>{proxy.once('error',no);proxy.listen(30091,'127.0.0.1',yes);});
await fs.mkdir('/scratch/work',{recursive:true});await fs.mkdir('/scratch/pi',{recursive:true});
await fs.cp('/input/engine','/scratch/work',{recursive:true,force:false,errorOnExist:true});
try{await fs.cp('/input/harness','/scratch/harness',{recursive:true,force:false,errorOnExist:true});}
catch(e){if(e.code!=='ENOENT')throw e;await fs.mkdir('/scratch/harness');}
const settings = {compaction: {enabled: false}, retry: {enabled: false, provider: {maxRetries: 0}}};
if (spec.provider_timeout_ms) {
  settings.httpIdleTimeoutMs = spec.provider_timeout_ms;
  settings.retry.provider.timeoutMs = spec.provider_timeout_ms;
}
await fs.writeFile('/scratch/pi/settings.json', JSON.stringify(settings));
await fs.writeFile('/scratch/pi/models.json',JSON.stringify({providers:{wire:{baseUrl:'http://127.0.0.1:30091/v1',api:'openai-completions',apiKey:spec.token,models:[spec.profile]}}}));
const args=['/opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js','--mode','json','--provider','wire','--model','deepseek-flash','--thinking','max','--no-session','--no-extensions','--no-mcp','--no-context-files','--no-skills','--no-prompt-templates'];
if(spec.tools)args.push('--tools',spec.tools.join(','));
args.push('--system-prompt',spec.system);
for(const e of spec.extensions)args.push('-e',`/input/harness/${e}`);
for(const e of spec.skills)args.push('--skill',`/input/harness/${e}`);
for(const e of spec.templates)args.push('--prompt-template',`/input/harness/${e}`);
if(spec.rpc)args.push('-e','/opt/ee/v3_rpc.mjs');
if(spec.paged_read)args.push('-e','/opt/ee/v3_paged_read.mjs');
args.push('--print',spec.prompt);
const child=spawn(process.execPath,args,{cwd:'/scratch/work',stdio:['ignore','inherit','inherit'],env:{PATH:process.env.PATH,HOME:'/scratch',USERPROFILE:'/scratch',PI_CODING_AGENT_DIR:'/scratch/pi',PI_OFFLINE:'1',PI_SKIP_VERSION_CHECK:'1',NO_PROXY:'127.0.0.1',EE_TOY_GATE:'http://127.0.0.1:30091'}});
const code=await new Promise((yes,no)=>{child.on('error',no);child.on('exit',(r)=>yes(r??1));});
for(const s of sockets)s.destroy();proxy.close();process.exitCode=code;
