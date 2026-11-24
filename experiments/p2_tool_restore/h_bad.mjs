// Fixed toy fixture: positive inputs work; intentionally fails for zero hits.
// Network traffic is restricted by this fixture to its per-session loopback witness.
export default function (pi) {
  const base = new URL(process.env.EE_FIXTURE_GATE);
  if (base.hostname !== '127.0.0.1' || base.protocol !== 'http:') throw Error('invalid fixture gateway');
  async function witness(kind, payload) {
    const res = await fetch(new URL('/fixture/' + kind, base), {
      method: 'POST', headers: {'Authorization': 'Bearer ' + process.env.EE_FIXTURE_LEASE, 'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw Error('fixture gateway rejected event');
    return (await res.json()).challenge;
  }
  pi.on('session_start', async () => { await witness('loaded', {fixture:'count-hits-bad-v1'}); });
  pi.registerTool({name:'count_hits',label:'Count exact hits',description:'Count literal hit entries in a toy array.',
    parameters:{type:'object',properties:{items:{type:'array',items:{type:'string'}}},required:['items']},
    async execute(id, args) {
      const n = args.items.filter(x => x === 'hit').length;
      const receipt = await witness('invoked', {id, n});
      if (n === 0) throw Error('TOY_ZERO_BUG:' + receipt);
      return {content:[{type:'text',text:'COUNT=' + n + ':' + receipt}],details:{count:n}};
    }
  });
}
