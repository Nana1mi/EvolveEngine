// Synthetic fixture only. Identical one-file read/edit scope as the old calibration;
// diagnostic rejection codes deliberately exclude paths, prompt text, and edit contents.
import {appendFileSync, realpathSync, statSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
export default function (pi) {
  const c=JSON.parse(process.env.EVOLVE_PI_GUARD); const file='metric.py';
  if(c.edit?.length!==1||c.read?.length!==1||c.edit[0]!==file||c.read[0]!==file||c.compile!==null) throw Error('bad fixture');
  const target=resolve(realpathSync(c.root),file); let ordinal=0;
  pi.on('session_start',()=>writeFileSync(c.ready,'ready\n'));
  pi.on('tool_call',({toolName:tool,input})=>{
    ++ordinal; let why='ok';
    if(ordinal>c.max_tools) why='lease';
    else if(tool!=='read'&&tool!=='edit') why='tool';
    else if(input?.path!==file) why='path_must_be_literal_metric.py';
    else if(tool==='read'&&(!Number.isSafeInteger(input.offset)||input.offset<1||!Number.isSafeInteger(input.limit)||input.limit<1||input.limit>c.max_lines)) why='read_needs_numeric_offset_and_bounded_limit';
    else {try{if(realpathSync(target)!==target||!statSync(target).isFile()) why='target_identity';}catch{why='target_missing';}}
    const accepted=why==='ok'; appendFileSync(c.receipt,JSON.stringify({tool,accepted,ordinal,why})+'\n');
    if(!accepted) return {block:true,reason:why};
  });
}
