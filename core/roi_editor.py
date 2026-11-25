"""Browser-side rectangle editor used by the Gradio ROI preview."""

HTML_TEMPLATE = """
<div class="roi-stage" style="display:${value.src ? 'inline-block' : 'none'};width:${value.display_width || 0}px">
  <img class="roi-photo" src="${value.src}" alt="Khung hình video để chọn ROI" draggable="false">
  <div class="roi-box" style="left:${value.roi[0]*100}%;top:${value.roi[1]*100}%;width:${(value.roi[2]-value.roi[0])*100}%;height:${(value.roi[3]-value.roi[1])*100}%">
    <span class="roi-handle nw" data-handle="nw"></span>
    <span class="roi-handle n" data-handle="n"></span>
    <span class="roi-handle ne" data-handle="ne"></span>
    <span class="roi-handle e" data-handle="e"></span>
    <span class="roi-handle se" data-handle="se"></span>
    <span class="roi-handle s" data-handle="s"></span>
    <span class="roi-handle sw" data-handle="sw"></span>
    <span class="roi-handle w" data-handle="w"></span>
  </div>
</div>
<div class="roi-empty" style="display:${value.src ? 'none' : 'block'}">Bấm “Xem trước ROI” để chọn vùng trên ảnh.</div>
"""

CSS_TEMPLATE = """
.roi-stage { position: relative; max-width: 100%; line-height: 0; user-select: none; touch-action: none; }
.roi-photo { display: block; width: 100%; height: auto; }
.roi-box { position: absolute; box-sizing: border-box; border: 3px solid #1fe45e;
  background: rgba(31, 228, 94, .10); cursor: move; touch-action: none; }
.roi-handle { position: absolute; width: 13px; height: 13px; border: 2px solid #fff;
  border-radius: 3px; background: #0baf3d; box-sizing: border-box; }
.roi-handle.nw { left: -7px; top: -7px; cursor: nwse-resize; }
.roi-handle.n  { left: calc(50% - 7px); top: -7px; cursor: ns-resize; }
.roi-handle.ne { right: -7px; top: -7px; cursor: nesw-resize; }
.roi-handle.e  { right: -7px; top: calc(50% - 7px); cursor: ew-resize; }
.roi-handle.se { right: -7px; bottom: -7px; cursor: nwse-resize; }
.roi-handle.s  { left: calc(50% - 7px); bottom: -7px; cursor: ns-resize; }
.roi-handle.sw { left: -7px; bottom: -7px; cursor: nesw-resize; }
.roi-handle.w  { left: -7px; top: calc(50% - 7px); cursor: ew-resize; }
.roi-empty { padding: 2rem; text-align: center; color: #777; border: 1px dashed #aaa; }
"""

JS_ON_LOAD = """
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
let drag = null;
function paint(box, roi) {
  box.style.left = `${roi[0] * 100}%`;
  box.style.top = `${roi[1] * 100}%`;
  box.style.width = `${(roi[2] - roi[0]) * 100}%`;
  box.style.height = `${(roi[3] - roi[1]) * 100}%`;
}
element.addEventListener('pointerdown', (event) => {
  const box = event.target.closest('.roi-box');
  if (!box || !element.contains(box) || !props.value?.src) return;
  const stage = box.parentElement;
  const rect = stage.getBoundingClientRect();
  drag = {
    box, stage, pointer: event.pointerId, handle: event.target.dataset.handle || 'move',
    x: event.clientX, y: event.clientY, width: rect.width, height: rect.height,
    start: [...props.value.roi], roi: [...props.value.roi]
  };
  stage.setPointerCapture(event.pointerId);
  event.preventDefault();
});
element.addEventListener('pointermove', (event) => {
  if (!drag || event.pointerId !== drag.pointer) return;
  const dx = (event.clientX - drag.x) / drag.width;
  const dy = (event.clientY - drag.y) / drag.height;
  const [x1, y1, x2, y2] = drag.start;
  const r = [...drag.start];
  const minSize = 0.02;
  if (drag.handle === 'move') {
    const shiftX = clamp(dx, -x1, 1 - x2);
    const shiftY = clamp(dy, -y1, 1 - y2);
    r[0] = x1 + shiftX; r[2] = x2 + shiftX;
    r[1] = y1 + shiftY; r[3] = y2 + shiftY;
  } else {
    if (drag.handle.includes('w')) r[0] = clamp(x1 + dx, 0, x2 - minSize);
    if (drag.handle.includes('e')) r[2] = clamp(x2 + dx, x1 + minSize, 1);
    if (drag.handle.includes('n')) r[1] = clamp(y1 + dy, 0, y2 - minSize);
    if (drag.handle.includes('s')) r[3] = clamp(y2 + dy, y1 + minSize, 1);
  }
  drag.roi = r;
  paint(drag.box, r);
});
function finish(event) {
  if (!drag || event.pointerId !== drag.pointer) return;
  const roi = drag.roi.map(v => Math.round(v * 100) / 100);
  drag = null;
  props.value = {...props.value, roi};
  trigger('submit');
}
element.addEventListener('pointerup', finish);
element.addEventListener('pointercancel', finish);
"""
