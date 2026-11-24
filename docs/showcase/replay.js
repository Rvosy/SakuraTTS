const movie = document.querySelector('#petMovie');
const seek = document.querySelector('#takeSeek');
const takePlay = document.querySelector('#takePlay');
const logList = document.querySelector('#eventLog');
const takeClock = document.querySelector('#takeTime');
const records = recordedTake.records;
const hasMovie = Boolean(recordedTake.video);
let takeDuration = recordedTake.duration;
const firstWake = records.find(record => record.cue === 'wake')?.at ?? 0;
let takePosition = hasMovie ? 0 : Math.max(0, firstWake - .6);
let previewOpened = false;
let previewOrigin = null;
let replayFrame;

document.querySelector('#takeKind').textContent = hasMovie ? '实录回放' : '历史日志回放 · 桌宠构图预览';
document.querySelector('#takeNote').textContent = hasMovie
  ? '终端保留原始日志的行序和时间。时序图随录屏同步。'
  : '对照等待回复和模型加载的色带，可以看到两者重叠的时间。桌宠日志精确到秒；右侧为录制构图预览。';
seek.max = takeDuration;

const chart = document.querySelector('#takeChart');
const rails = ['大模型请求', '语音引擎', '播放端', '服务状态'].map(label => {
  const row = document.createElement('div');
  row.className = 'take-lane';
  const title = document.createElement('span');
  title.textContent = label;
  const rail = document.createElement('div');
  rail.className = 'take-rail';
  row.append(title, rail);
  chart.append(row);
  return rail;
});
const cursor = document.createElement('div');
cursor.className = 'take-cursor';
chart.append(cursor);
const marks = [];
function addBand(lane, at, label, color) {
  const bar = document.createElement('i');
  bar.className = color + '-band';
  bar.title = label;
  const mark = {bar, start: at, end: at, label};
  rails[lane].append(bar);
  marks.push(mark);
  return mark;
}

const pending = new Map();
const nativeSynthesis = records.some(record => record.cue === 'synth-start');
let status;
function setStatus(at, label, color) {
  if (status?.label === label) return;
  if (status) status.end = at;
  status = addBand(3, at, label, color);
  status.end = recordedTake.duration;
}
for (const {at, cue, request} of records) {
  if (!cue) continue;
  if (cue === 'sleeping') setStatus(at, '休眠', 'sleep');
  if (cue === 'loading') setStatus(at, '加载中', 'load');
  if (cue === 'awake') setStatus(at, '已加载', 'run');
  if (cue === 'releasing') setStatus(at, '释放模型', 'load');
  if (cue === 'wake') setStatus(at, '唤醒', 'load');
  const phase = {
    'llm-start': [0, '等待回复', 'ask'], 'loading': [1, '加载', 'load'],
    'synth-start': [1, '合成', 'run'], 'job-start': [1, '合成请求', 'run'],
    'play-start': [2, '播放', 'play']
  }[cue];
  if (nativeSynthesis && cue.startsWith('job-')) continue;
  const key = cue.replace(/-(start|end)$/, '') + (request ?? '');
  if (phase) {
    if (!pending.has(key)) pending.set(key, addBand(phase[0], at, phase[1], phase[2]));
    if (cue === 'synth-start' && status) setStatus(at, '合成中', 'run');
  } else {
    const endKey = cue === 'awake' ? 'loading' : key;
    if ((cue.endsWith('-end') || cue === 'awake') && pending.has(endKey)) {
      pending.get(endKey).end = at;
      pending.delete(endKey);
      if (['synth-end', 'job-end'].includes(cue) && status && !status.label.includes('休眠')) {
        setStatus(at, '空闲驻留', 'idle');
      }
    }
  }
}

const playEnd = records.findLast(record => record.cue === 'play-end')?.at;
const hasIdleTail = playEnd > 0 && records.some(record => record.at > playEnd && record.cue === 'sleeping')
  && !records.some(record => record.at > playEnd && ['llm-start', 'wake', 'loading', 'job-start', 'synth-start', 'play-start'].includes(record.cue));
const states = marks.filter(mark => mark.bar.parentElement === rails[3]);
const chartTitle = document.querySelector('#chartTitle');
const chartState = document.querySelector('#chartState');
const rawButton = document.querySelector('#rawLog');
let fullLog = false;
let shown = 0;

function terminalLine(record) {
  const match = record.raw.match(/^(?:\d{4}-\d{2}-\d{2} (\d\d:\d\d:\d\d),\d+\s+(INFO|WARNING|ERROR|CRITICAL) \[(sakuratts\.[^\]]+)\] \[请求 [^\]]+\] |\[(\d\d:\d\d:\d\d)\] \[[^\]]+\] \[(?:info|error|warning)\] )(.+)/i);
  if (!match) return null;
  const message = match[5].split(' │ ')[0];
  if (match[4] && !record.cue) return null;
  return {time: match[1] ?? match[4], message};
}
const terminal = records.map(terminalLine);
const baseDpr = window.devicePixelRatio;
function sizeTerminal() {
  logList.style.fontSize = (13 * baseDpr / window.devicePixelRatio) + 'px';
}
sizeTerminal();
window.addEventListener('resize', sizeTerminal);

function paintLog() {
  if (shown && records[shown - 1].at > takePosition + .000001) {
    shown = 0;
    logList.replaceChildren();
  }
  const batch = document.createDocumentFragment();
  while (shown < records.length && records[shown].at <= takePosition + .000001) {
    const record = records[shown];
    const line = terminal[shown++];
    if (fullLog) batch.append(record.raw + '\n');
    else if (line) {
      const time = document.createElement('time');
      time.textContent = line.time;
      batch.append(time, '  ' + line.message + '\n');
    }
  }
  if (batch.hasChildNodes()) {
    logList.append(batch);
    logList.scrollTop = logList.scrollHeight;
  }
}
rawButton.onclick = () => {
  fullLog = !fullLog;
  rawButton.setAttribute('aria-pressed', String(fullLog));
  shown = 0;
  logList.replaceChildren();
  logList.scrollLeft = 0;
  paintLog();
};

function paintTake(seconds) {
  takePosition = Math.max(0, Math.min(seconds, takeDuration));
  seek.value = takePosition;
  takeClock.textContent = takePosition.toFixed(2) + ' / ' + takeDuration.toFixed(2) + ' s';
  paintLog();

  const tail = hasIdleTail && takePosition >= playEnd;
  const from = tail ? playEnd : 0;
  const until = hasIdleTail && !tail ? playEnd : takeDuration;
  const span = until - from;
  chartTitle.textContent = tail ? '播放后' : '对话时序';
  const currentState = states.findLast(mark => mark.start <= takePosition);
  chartState.textContent = currentState ? '引擎 · ' + currentState.label : '';
  rails.forEach((rail, i) => { rail.parentElement.hidden = tail ? i !== 3 : i === 3; });
  chart.style.setProperty('--progress', (takePosition - from) / span);
  for (const {bar, start, end, label} of marks) {
    const left = Math.max(from, start);
    bar.hidden = start > until || end < from || takePosition < left;
    if (bar.hidden) continue;
    const width = Math.max(0, Math.min(end, takePosition, until) - left) / span;
    bar.style.left = ((left - from) / span * 100) + '%';
    bar.style.width = width ? (width * 100) + '%' : '2px';
    bar.textContent = width > label.length * .014 ? label : '';
  }
  takePlay.textContent = (hasMovie ? !movie.paused : previewOrigin !== null) ? '暂停回放'
    : takePosition >= takeDuration ? '重播' : '播放回放';
}

function runFrame() {
  const seconds = hasMovie ? movie.currentTime : (performance.now() - previewOrigin) / 1000;
  if (!hasMovie && seconds >= takeDuration) previewOrigin = null;
  paintTake(seconds);
  if (hasMovie ? !movie.paused : previewOrigin !== null) replayFrame = requestAnimationFrame(runFrame);
}

function pauseTake() {
  movie.pause();
  previewOrigin = null;
  cancelAnimationFrame(replayFrame);
  paintTake(hasMovie ? movie.currentTime : takePosition);
}

function openTake() {
  if (hasMovie || previewOpened) return;
  previewOpened = true;
  previewOrigin = performance.now() - takePosition * 1000;
  runFrame();
}

takePlay.onclick = async () => {
  if (hasMovie) {
    if (!movie.paused) return pauseTake();
    if (movie.ended) movie.currentTime = 0;
    try {
      await movie.play();
    } catch (error) {
      if (error.name !== 'AbortError') takeClock.textContent = '录屏无法播放';
    }
  } else if (previewOrigin !== null) {
    pauseTake();
  } else {
    previewOrigin = performance.now() - (takePosition >= takeDuration ? 0 : takePosition * 1000);
    runFrame();
  }
};

seek.oninput = () => {
  const seconds = Number(seek.value);
  if (hasMovie) movie.currentTime = seconds;
  if (previewOrigin !== null) previewOrigin = performance.now() - seconds * 1000;
  paintTake(seconds);
};

if (hasMovie) {
  movie.hidden = false;
  document.querySelector('#petStill').hidden = true;
  movie.onloadedmetadata = () => {
    takeDuration = movie.duration;
    seek.max = takeDuration;
    paintTake(movie.currentTime);
  };
  movie.onplay = () => { cancelAnimationFrame(replayFrame); runFrame(); };
  movie.onpause = movie.onseeked = movie.onended = () => paintTake(movie.currentTime);
  movie.onerror = () => { takeClock.textContent = '无法读取桌宠录屏'; takePlay.disabled = true; };
  movie.src = recordedTake.video;
}
paintTake(takePosition);
