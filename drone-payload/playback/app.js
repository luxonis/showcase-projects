"use strict";
let recording, telemetry=[], videos=[], playing=false, position=0, duration=0, speed=1, rangeStart=0, rangeEnd=0;
const byId=id=>document.getElementById(id);
const formatTime=value=>{const ms=Math.max(0,Math.round(value*1000));return `${Math.floor(ms/60000)}:${(ms%60000/1000).toFixed(3).padStart(6,"0")}`;};
const graphSpecs=[['height-chart','height_m','#83d0cb'],['pressure-chart','pressure_pa','#81b6f5'],['temperature-chart','temperature_c','#e6b387']];

function parseTime(value){
  const text=value.trim();
  if(!/^(?:\d+:){0,2}\d+(?:\.\d+)?$/.test(text))return NaN;
  const parts=text.split(':').map(Number);
  if((parts.length>1&&parts.at(-1)>=60)||(parts.length===3&&parts[1]>=60))return NaN;
  return parts.reduce((total,part)=>total*60+part,0);
}
function setRange(start,end){
  playing=false;byId('play').textContent='Play all cameras';
  rangeStart=start;rangeEnd=end;
  byId('timeline').min=start;byId('timeline').max=end;
  byId('range-start').value=formatTime(start);byId('range-end').value=formatTime(end);
  byId('range-start').removeAttribute('aria-invalid');byId('range-end').removeAttribute('aria-invalid');
  byId('range-error').textContent='';
  byId('range-summary').textContent=`${formatTime(end-start)} selected · ${formatTime(duration)} total`;
  seek(start);
}
function applyRange(event){
  event.preventDefault();
  const start=parseTime(byId('range-start').value),end=parseTime(byId('range-end').value);
  const validStart=Number.isFinite(start)&&start>=0&&start<duration;
  // Inputs display milliseconds, while recording duration can be more precise.
  const validEnd=Number.isFinite(end)&&end>start&&end<=duration+.0005;
  byId('range-start').setAttribute('aria-invalid',String(!validStart));
  byId('range-end').setAttribute('aria-invalid',String(!validEnd));
  if(!validStart||!validEnd){
    byId('range-error').textContent=`Enter a start before the end, between 0 and ${formatTime(duration)}. Use seconds or m:ss.sss.`;
    byId(validStart?'range-end':'range-start').focus();return;
  }
  setRange(start,Math.min(end,duration));
}

function fitCameraGrid(){
  if(!videos.length)return;
  const grid=byId('cameras'),styles=getComputedStyle(grid),gap=parseFloat(styles.gap)||0;
  const chrome=Math.max(...videos.map(({element})=>{
    const card=element.parentElement;
    return card.querySelector('header').getBoundingClientRect().height+card.querySelector('.details').getBoundingClientRect().height+2;
  }));
  let bestColumns=1,bestArea=-1;
  for(let columns=1;columns<=videos.length;columns++){
    const rows=Math.ceil(videos.length/columns);
    const width=(grid.clientWidth-gap*(columns-1))/columns;
    const height=Math.max(0,(grid.clientHeight-gap*(rows-1))/rows-chrome);
    const area=videos.reduce((total,{info})=>{
      const scale=Math.min(width/info.width,height/info.height);
      return total+info.width*info.height*scale*scale;
    },0);
    if(area>bestArea){bestArea=area;bestColumns=columns;}
  }
  grid.style.setProperty('--camera-columns',bestColumns);
  grid.style.setProperty('--camera-rows',Math.ceil(videos.length/bestColumns));
}
function graphPadding(width,height){
  return {left:Math.min(62,Math.max(38,width*.23)),right:width<240?8:12,top:height<100?6:12,bottom:height<100?18:26};
}

function sensorAt(time){
  let left=0,right=telemetry.length;
  while(left<right){const mid=(left+right)>>1;if(telemetry[mid].time_us/1e6<=time)left=mid+1;else right=mid;}
  const row=left&&telemetry[left-1].time_us/1e6>=rangeStart?telemetry[left-1]:telemetry[left];
  return row&&row.time_us/1e6>=rangeStart&&row.time_us/1e6<=rangeEnd?row:null;
}
function drawGraph(id,key,color){
  const canvas=byId(id),ratio=window.devicePixelRatio||1;
  const width=canvas.clientWidth,height=canvas.clientHeight;
  canvas.width=width*ratio;canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const points=telemetry.filter(row=>row.time_us/1e6>=rangeStart&&row.time_us/1e6<=rangeEnd&&Number.isFinite(row[key]));
  if(!points.length){ctx.fillStyle='#8ca4b5';ctx.fillText('No sensor samples',12,30);return;}
  let min=Infinity,max=-Infinity;for(const row of points){min=Math.min(min,row[key]);max=Math.max(max,row[key]);}
  const padding=Math.max((max-min)*.1,key==='height_m'?.25:key==='pressure_pa'?1:.05);min-=padding;max+=padding;
  const {left,right,top,bottom}=graphPadding(width,height),plotWidth=width-left-right,plotHeight=height-top-bottom;
  const x=time=>left+(time-rangeStart)/Math.max(rangeEnd-rangeStart,.001)*plotWidth;
  const y=value=>top+(max-value)/(max-min)*plotHeight;
  ctx.font=width<240?'9px system-ui':'11px system-ui';ctx.lineWidth=1;
  const ticks=height<100?2:4;
  for(let n=0;n<=ticks;n++){
    const value=min+(max-min)*n/ticks,py=y(value);ctx.strokeStyle='#293b47';ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(width-right,py);ctx.stroke();
    ctx.fillStyle='#8ca4b5';ctx.textAlign='right';ctx.fillText(value.toFixed(key==='pressure_pa'?0:2),left-8,py+4);
  }
  ctx.textAlign='left';ctx.fillText(formatTime(rangeStart),left,height-6,Math.max(1,plotWidth/2-4));
  ctx.textAlign='right';ctx.fillText(formatTime(rangeEnd),width-right,height-6,Math.max(1,plotWidth/2-4));
  ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
  const stride=Math.max(1,Math.floor(points.length/Math.max(plotWidth*2,1)));
  for(let n=0;n<points.length;n+=stride){const row=points[n];n?ctx.lineTo(x(row.time_us/1e6),y(row[key])):ctx.moveTo(x(row.time_us/1e6),y(row[key]));}
  ctx.stroke();ctx.strokeStyle='#edf5fa';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(position),top);ctx.lineTo(x(position),height-bottom);ctx.stroke();
}
function refresh(){
  byId('timeline').value=position;byId('clock').textContent=`${formatTime(position)} / ${formatTime(rangeEnd)}`;
  const row=sensorAt(position);
  byId('height').textContent=Number.isFinite(row?.height_m)?`${row.height_m.toFixed(2)} m`:'—';
  byId('pressure').textContent=Number.isFinite(row?.pressure_pa)?`${row.pressure_pa.toFixed(1)} Pa`:'—';
  byId('temperature').textContent=Number.isFinite(row?.temperature_c)?`${row.temperature_c.toFixed(2)} °C`:'—';
  for(const spec of graphSpecs)drawGraph(...spec);
}
function syncVideos(force=false){
  for(const {element,info} of videos){
    const active=position>=info.offset_s&&position<info.end_s;
    const desired=Math.max(0,Math.min(position-info.offset_s,element.duration||Infinity));
    if(Number.isFinite(element.duration)&&(force||Math.abs(element.currentTime-desired)>.12))element.currentTime=desired;
    element.playbackRate=speed;
    if(playing&&active&&!element.ended){element.play().catch(error=>{if(error.name!=='AbortError')byId('notice').textContent=`Playback: ${error.message}`;});}else element.pause();
    element.style.opacity=active?'1':'.45';
  }
}
function seek(time){position=Math.max(rangeStart,Math.min(time,rangeEnd));syncVideos(true);refresh();}
function toggle(){
  if(position>=rangeEnd)seek(rangeStart);playing=!playing;
  byId('play').textContent=playing?'Pause all cameras':'Play all cameras';syncVideos();
}
let previous=performance.now(),lastDraw=0;
function tick(now){
  if(playing){position=Math.min(rangeEnd,position+(now-previous)/1000*speed);const ended=position>=rangeEnd;if(ended){playing=false;byId('play').textContent='Play all cameras';}syncVideos(ended);}
  previous=now;if(now-lastDraw>60){refresh();lastDraw=now;}requestAnimationFrame(tick);
}
async function load(){
  const responses=await Promise.all([fetch('/api/playback'),fetch('/api/telemetry')]);
  if(responses.some(response=>!response.ok))throw new Error('Unable to load the recording');
  [recording,telemetry]=await Promise.all(responses.map(response=>response.json()));
  duration=recording.duration_s;rangeEnd=duration;
  const sessionName=recording.manifest.session_number?`Session ${String(recording.manifest.session_number).padStart(6,'0')}`:recording.manifest.id;
  byId('summary').textContent=`${sessionName} · ${recording.manifest.status} · ${recording.cameras.length} camera streams · ${telemetry.length.toLocaleString()} sensor samples`;
  let notices=[...recording.warnings];
  if(!recording.cameras.length)notices.push('No camera footage is available in this recording.');
  for(const info of recording.cameras){
    const card=document.createElement('article');card.className='camera';
    const header=document.createElement('header'),title=document.createElement('strong'),detail=document.createElement('span');
    title.textContent=info.name;detail.textContent=`${info.sensor} · ${info.width}×${info.height} · ${info.fps} FPS`;detail.title=detail.textContent;
    header.append(title,detail);
    const element=document.createElement('video');element.src=info.url;element.preload='metadata';element.muted=true;element.playsInline=true;element.controls=false;
    element.addEventListener('loadedmetadata',()=>{if(element.videoWidth!==info.width||element.videoHeight!==info.height)byId('notice').textContent='Video dimensions differ from the recording manifest';syncVideos(true);});
    element.addEventListener('error',()=>{byId('notice').textContent=`${info.name}: the browser could not decode this video.`;});
    element.addEventListener('dblclick',()=>element.requestFullscreen());
    const details=document.createElement('div');details.className='details';details.textContent=`${info.frames.toLocaleString()} frames · ${info.missing_frames} detected missing frames · Double-click for fullscreen`;details.title=details.textContent;
    card.append(header,element,details);byId('cameras').append(card);videos.push({element,info});
  }
  byId('notice').textContent=notices.join(' ');byId('play').disabled=duration<=0;
  for(const id of ['range-start','range-end','apply-range','full-range'])byId(id).disabled=duration<=0;
  setRange(0,duration);fitCameraGrid();requestAnimationFrame(tick);
}
byId('play').addEventListener('click',toggle);byId('reset').addEventListener('click',()=>seek(rangeStart));
byId('range-form').addEventListener('submit',applyRange);
byId('full-range').addEventListener('click',()=>setRange(0,duration));
byId('timeline').addEventListener('input',event=>seek(Number(event.target.value)));
byId('speed').addEventListener('change',event=>{speed=Number(event.target.value);syncVideos();});
for(const [id] of graphSpecs)byId(id).addEventListener('click',event=>{const bounds=event.target.getBoundingClientRect(),padding=graphPadding(bounds.width,bounds.height);seek(rangeStart+(event.clientX-bounds.left-padding.left)/(bounds.width-padding.left-padding.right)*(rangeEnd-rangeStart));});
new ResizeObserver(()=>{fitCameraGrid();refresh();}).observe(byId('cameras'));
window.addEventListener('resize',()=>{fitCameraGrid();refresh();});load().catch(error=>{byId('notice').textContent=error.message;});
