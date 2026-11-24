const stage = document.querySelector('#stage');
const pages = [...stage.querySelectorAll('section')];
const memory = document.querySelector('.memory');
const wake = document.querySelector('.wake');
const live = document.querySelector('.live');
const staging = document.querySelector('.staging');
const tierPanels = [...staging.querySelectorAll('.stage-explanation [data-tier]')];
const builds = new Map([[wake, 2], [memory, 2], [staging, tierPanels.length]]);
const steps = pages.flatMap(page => Array(builds.get(page) ?? 1).fill(page));
const play = document.querySelector('#play');
const notes = document.querySelector('#notes');
const previous = document.querySelector('#prev');
const next = document.querySelector('#next');
let position = 0;
let playing = false;
let timer;
let controlsTimer;

const profiles = [
  {label: 'FP32', original: true, peak: 3931.7, idle: 3076.0, time: 5.548},
  {label: 'FP16', original: true, peak: 2323.2, idle: 1677.2, time: 4.407},
  {label: 'FP32 全精度', peak: 2148.6, idle: 1253.1, time: 3.290},
  {label: 'FP16 标准档 · 默认', peak: 758.2, idle: 569.4, time: 2.677},
  {label: 'FP16 低显存档', peak: 584.1, idle: 387.0, time: 3.560},
  {label: 'FP16 极限档', peak: 465.1, idle: 106.5, time: 5.481}
];

const timeBars = document.querySelector('#timeBars');
const mbFormat = new Intl.NumberFormat('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
const peakScale = Math.max(...profiles.map(sample => sample.peak));
const timeScale = Math.max(...profiles.map(sample => sample.time));

for (const sample of profiles) {
  const label = (sample.original ? '原版 · ' : 'SakuraTTS · ') + sample.label;
  const memoryRow = document.createElement('div');
  memoryRow.className = 'bench-row';
  memoryRow.innerHTML = '<span class="bench-label"></span><div class="bench-track"><i class="peak"><i class="idle-layer"></i></i></div><span class="peak-value"></span><span class="resident-value idle-only"></span>';
  memoryRow.querySelector('.bench-label').textContent = label;
  memoryRow.querySelector('.peak').style.width = (100 * sample.peak / peakScale) + '%';
  memoryRow.querySelector('.idle-layer').style.width = (100 * sample.idle / sample.peak) + '%';
  memoryRow.querySelector('.peak-value').textContent = mbFormat.format(sample.peak);
  memoryRow.querySelector('.resident-value').textContent = mbFormat.format(sample.idle);
  document.querySelector('#memoryBars').append(memoryRow);

  const row = document.createElement('div');
  row.className = sample.original ? 'bench-row original' : 'bench-row';
  row.innerHTML = '<span class="bench-label"></span><div class="bench-track"><i></i></div><b class="reading"></b>';
  row.querySelector('.bench-label').textContent = label;
  row.querySelector('i').style.width = (100 * sample.time / timeScale) + '%';
  row.querySelector('.reading').textContent = sample.time.toFixed(3) + ' 秒';
  timeBars.append(row);
}

let conversationAnimations = [];

function stopConversation() {
  for (const animation of conversationAnimations) animation.cancel();
  conversationAnimations = [];
}

function animateConversation(early) {
  stopConversation();
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const offset = early ? 0.14 : 0;
  const requestDuration = 0.08;
  const timing = {duration: 7000, delay: early ? 750 : 350, iterations: Infinity, fill: 'both'};
  const phases = [
    ['.ask', offset, requestDuration], ['.reply', offset + requestDuration, 0.24], ['.synth', offset + 0.32, 0.13],
    ['.listen', offset + 0.45, 0.13]
  ];
  if (early) phases.push(['.prepare', offset, 0.19]);
  for (const [selector, start, duration] of phases) {
    const animation = wake.querySelector(selector).animate([
      {transform: 'scaleX(0)', offset: 0},
      {transform: 'scaleX(0)', offset: start},
      {transform: 'scaleX(1)', offset: start + duration},
      {transform: 'scaleX(1)', offset: 1}
    ], {...timing, pseudoElement: '::before'});
    conversationAnimations.push(animation);
  }
  if (early) {
    const arrow = wake.querySelector('.wake-arrow').animate({
      clipPath: ['inset(0 0 100% -.5cqw)', 'inset(0 0 100% -.5cqw)', 'inset(0 0 0% -.5cqw)', 'inset(0 0 0% -.5cqw)'],
      offset: [0, offset, offset + requestDuration, 1]
    }, timing);
    conversationAnimations.push(arrow);
  }
}

function showControls() {
  stage.classList.add('controls');
  clearTimeout(controlsTimer);
  controlsTimer = setTimeout(() => stage.classList.remove('controls'), 1800);
}

function schedule() {
  clearTimeout(timer);
  const seconds = steps[position].dataset.seconds;
  if (!seconds) playing = false;
  play.disabled = !seconds;
  play.textContent = playing ? '暂停' : '自动播放';
  play.setAttribute('aria-pressed', String(playing));
  if (playing) timer = setTimeout(() => move(1), Number(seconds) * 1000);
}

function render() {
  const current = steps[position];
  if (current !== live && live.classList.contains('active')) pauseTake();
  if (current === live && !live.classList.contains('active')) openTake();
  for (const page of pages) {
    page.classList.toggle('active', page === current);
    page.inert = page !== current;
  }
  if (current === wake) {
    const build = position - steps.indexOf(wake);
    const early = build > 0;
    wake.classList.toggle('early', early);
    animateConversation(early);
    document.querySelector('#wakeLine').textContent = ['模型常驻', 'SakuraTTS 按需唤醒'][build];
    document.querySelector('#wakeCaption').textContent = [
      '发消息前，语音模型已经在显存里。第一句回复一到就能合成，播完后模型也继续留在显存里。',
      '主程序发出大模型请求时，也向语音引擎发送唤醒请求。引擎在等待回复期间加载模型。空闲与保活时间都到期后，服务释放推理占用的显存，回到休眠。',
    ][build];
    document.querySelector('#wakeNote').textContent = [
      '模型已经加载，大模型的回复一到就能开始合成。等待下一条消息时，模型仍然占用显存。',
      '图中以 managed 模式的标准档为例。如果文字先到，合成只等剩余的加载时间。休眠时推理进程退出，控制服务继续接收请求，下次唤醒时再启动推理进程。',
    ][build];
  }
  if (current !== wake && conversationAnimations.length) stopConversation();
  if (current === staging) {
    const panel = tierPanels[position - steps.indexOf(staging)];
    staging.dataset.tier = panel.dataset.tier;
    for (const tier of tierPanels) tier.hidden = tier !== panel;
  }
  if (current === memory) {
    const idle = steps[position - 1] === memory;
    memory.classList.toggle('show-idle', idle);
    for (const label of memory.querySelectorAll('.idle-only')) {
      label.setAttribute('aria-hidden', String(!idle));
    }
  }
  document.querySelector('#counter').textContent = (position + 1) + ' / ' + steps.length;
  previous.disabled = position === 0;
  next.disabled = position === steps.length - 1;
  schedule();
}

function move(delta) {
  const target = position + delta;
  if (target < 0 || target >= steps.length) return;
  position = target;
  render();
}

previous.onclick = () => move(-1);
next.onclick = () => move(1);
play.onclick = () => { playing = !playing; schedule(); };
notes.onclick = () => notes.setAttribute('aria-expanded', String(stage.classList.toggle('showNotes')));
stage.onpointermove = showControls;
stage.onclick = event => {
  if (!event.target.closest('a, button, footer, aside, .live')) move(1);
};
window.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.target.matches('input[type=range]')) return;
  if (event.key === ' ' && event.target.closest('button, a')) return;
  if (['ArrowLeft', 'ArrowRight', ' '].includes(event.key)) {
    event.preventDefault();
    move(event.key === 'ArrowLeft' ? -1 : 1);
  }
});
render();
showControls();
