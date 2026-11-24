// User-level bridge for Pi's HTTP client. All authority is in the other container.
import http from 'node:http';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';
const socketPath='/gate/gateway.sock';
const spec=JSON.parse(await fs.readFile('/input/config.json','utf8'));
const sockets=new Set();
const proxy=http.createServer((req,res)=>{
  const upstream=http.request({socketPath,path:req.url,method:req.method,headers:req.headers},(reply)=>{
    res.writeHead(reply.statusCode,reply.headers);reply.pipe(res);
  });
  upstream.on('error',()=>{res.statusCode=502;res.end('fixture gateway disconnected');});
  req.pipe(upstream);
});
proxy.on('connection',(s)=>{sockets.add(s);s.on('close',()=>sockets.delete(s));});
await new Promise((yes,no)=>{proxy.once('error',no);proxy.listen(30091,'127.0.0.1',yes);});
await fs.mkdir('/scratch/work',{recursive:true});await fs.mkdir('/scratch/pi',{recursive:true});
await fs.writeFile('/scratch/work/facts.txt',spec.facts);
await fs.writeFile('/scratch/work/engine.txt',spec.engine);
await fs.writeFile('/scratch/pi/settings.json',JSON.stringify({compaction:{enabled:false},retry:{enabled:false,provider:{maxRetries:0}}}));
await fs.writeFile('/scratch/pi/models.json',JSON.stringify({providers:{wire:{baseUrl:'http://127.0.0.1:30091/v1',api:'openai-completions',apiKey:'toy-only',models:[spec.profile]}}}));
const args=['/opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js','--mode','json','--provider','wire','--model','deepseek-flash','--thinking','max','--no-session','--no-extensions','--no-mcp','--no-context-files','--no-skills','--no-prompt-templates','--tools','read,write,fixture_ops,count_*','--system-prompt',spec.system,'-e','/opt/fixture/agent.mjs','--print',spec.prompt];
const child=spawn(process.execPath,args,{cwd:'/scratch/work',stdio:['ignore','inherit','inherit'],env:{PATH:process.env.PATH,HOME:'/scratch',USERPROFILE:'/scratch',PI_CODING_AGENT_DIR:'/scratch/pi',PI_OFFLINE:'1',PI_SKIP_VERSION_CHECK:'1',NO_PROXY:'127.0.0.1'}});
const code=await new Promise((yes,no)=>{child.on('error',no);child.on('exit',(r)=>yes(r??1));});
for(const s of sockets)s.destroy();proxy.close();process.exitCode=code;

