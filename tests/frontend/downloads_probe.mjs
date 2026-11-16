import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(process.argv[2],'utf8').match(/<script>([\s\S]*)<\/script>/)[1];
const entryMode=process.argv[3] || 'http';
function element(tag='div'){
  const queries=new Map();
  return {tagName:tag,children:[],value:'',checked:false,disabled:false,style:{},dataset:{},
    textContent:'',innerHTML:'',scrollTop:0,scrollHeight:0,clientHeight:0,
    classList:{add(){},remove(){}},
    appendChild(child){this.children.push(child);return child;},
    replaceChildren(...children){this.children=children;},
    addEventListener(){},removeEventListener(){},setAttribute(){},getAttribute(){return null;},
    remove(){},focus(){},scrollIntoView(){},
    querySelector(selector){if(!queries.has(selector))queries.set(selector,element());return queries.get(selector);},
    querySelectorAll(selector){if(selector==='.step'){if(!queries.has(selector))queries.set(selector,Array.from({length:5},()=>element()));return queries.get(selector);}return [];},
  };
}
const nodes=new Map();
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},
  createElement:element,createDocumentFragment:element,addEventListener(){},querySelector(){return element();},querySelectorAll(){return [];},body:element()};
const languages=[{value:'auto',label:'自动'},{value:'zh',label:'中文'},{value:'ja',label:'日语'},{value:'en',label:'英语'}];
let failDownload=false;
const requests=[];
const completed=id=>({id,kind:'download',name:'Test video',status:'done',stage:'完成',progress:1,logs:[],opts:{},
  files:[{name:id+'.mp4',path:'D:/downloads/'+id+'.mp4',kind:'视频'}],result:{media_paths:['D:/downloads/'+id+'.mp4']}});
const context=vm.createContext({document,window:{open(){},location:{protocol:entryMode==='file'?'file:':'http:'}},console,URL,Blob:class{},FormData:class{},alert(){},confirm(){return true;},
  setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,1)),clearTimeout,
  localStorage:{getItem:()=>JSON.stringify({backend:'qwen',language:'en'}),setItem(){}},
  fetch:async(url,opts)=>{
    requests.push({url:String(url),opts});let body={};let status=200;
    if(entryMode==='offline')throw new Error('offline test');
    if(url==='/api/models')body={asr:[],separation:[],diarize:[],qwen_languages:languages,download_ready:true,default_output:'D:/out'};
    if(url==='/api/jobs')body={jobs:[]};
    if(url==='/api/resumable')body={items:[]};
    if(url==='/api/download'){body=failDownload?{error:'测试平台需要登录'}:{id:'fresh',kind:'download',status:'wait'};if(failDownload)status=400;}
    if(url==='/api/run'){
      const {path}=JSON.parse(opts.body);
      if(path==='offline.wav') throw new Error('验证网络错误');
      if(path==='invalid.wav'){body={error:'验证路径错误'};status=400;}
      else body={id:'accepted-'+path};
    }
    if(String(url).startsWith('/api/status')){
      const id=new URL(url,'http://test').searchParams.get('id');
      body=id==='fresh'?completed('fresh'):{id,status:'done',stage:'完成',files:[],logs:[]};
    }
    return {ok:status===200,status,json:async()=>body,text:async()=>''};
  },
});
vm.runInContext(source+`;globalThis.P={setLanguages,addJob,updateJob,addFiles,addPendingPaths,baseReq,pending:()=>pending,jobs:()=>jobs,els:()=>els,catalog:()=>qwenLanguages};`,context);
const settle=()=>new Promise(resolve=>setTimeout(resolve,25));
await settle();
const $=id=>document.getElementById(id),P=context.P;
if(entryMode!=='http'){
  console.log(JSON.stringify({requests:requests.length,notice:$('connectionNotice').textContent,
    href:$('openWorkbench').href,disabled:$('go').disabled,catalog:P.catalog().length}));
  process.exit(0);
}
const restored=$('language').value;
P.setLanguages('auto');const auto=$('language').value;
$('backend').value='whisper';$('backend').onchange();const whisper=$('language').children.map(o=>o.value);
$('backend').value='qwen';$('backend').onchange();const qwen=$('language').children.map(o=>o.value);
$('shareLink').value='分享 https://example.com/watch?v=1';$('outDir').value='D:/chosen';$('cookiesFile').value='D:/cookies.txt';
await $('downloadLink').onclick();await settle();
const request=JSON.parse(requests.find(r=>r.url==='/api/download').opts.body);
const fresh=P.jobs().find(j=>j.id==='fresh');P.updateJob(fresh);P.updateJob(fresh);
const pendingAfterRepeatedUpdates=P.pending().length;
P.addJob(completed('history'));
const pendingBeforeManual=P.pending().length;
const history=P.els().history;
history.acts.children.find(x=>x.textContent==='加入待处理').onclick();
const manualCount=P.pending().length;
const binaryPreview=history.acts.children.some(x=>String(x.textContent).startsWith('预览'));
const historyButton=history.acts.children.find(x=>x.textContent==='已加入待处理');
$('files').children[1].children.at(-1).onclick();
const removedDownload=historyButton.textContent;
historyButton.onclick();
const readdedDownload=historyButton.textContent;
P.addJob({id:'audio',status:'done',stage:'完成',files:[{kind:'音频',name:'extracted.wav',path:'good.wav'}],logs:[]});
const audioBox=P.els().audio.acts.children.find(x=>x.className==='audiobox');
audioBox.children.find(x=>x.textContent==='用此结果继续处理').onclick();
const queuedAudio=P.pending().some(x=>x.path==='good.wav') && !requests.some(r=>r.url==='/api/run');
P.addPendingPaths(['invalid.wav','offline.wav']);
$('taskMode').value='separate';$('sepModel').value='MDX';$('doSep').checked=false;
await $('go').onclick();
const separateRequest=JSON.parse(requests.find(r=>r.url==='/api/run').opts.body);
const retained=P.pending().map(x=>[x.path,x.error]);
$('files').children[1].children[1].value='fixed.wav';
$('files').children[1].children[1].oninput();
await $('files').children[1].children.at(-1).onclick();
const remaining=P.pending().map(x=>x.path);
P.addFiles([{name:'duplicate.wav',size:1},{name:'duplicate.wav',size:2}]);
const duplicateUploadCount=P.pending().filter(x=>x.name==='duplicate.wav').length;
failDownload=true;await $('downloadLink').onclick();await settle();
console.log(JSON.stringify({restored,auto,whisper,qwen,request,pendingAfterRepeatedUpdates,pendingBeforeManual,manualCount,
  removedDownload,readdedDownload,queuedAudio,separateRequest,retained,remaining,duplicateUploadCount,
  binaryPreview,speakerDisplay:history.spkBtn.style.display,error:$('downloadNote').textContent,locked:$('downloadLink').disabled}));
