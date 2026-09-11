from pathlib import Path

import cv2
import numpy as np

from ai.render import draw_detections

_CUDA_CTX = None      # 프로세스 전체가 함께 쓰는 CUDA 컨텍스트
_CUDA_USERS = 0       # 이걸 쓰고 있는 디텍터 수


class JetsonTensorRTDetector:
    """
    Jetson TensorRT 직접 추론 버전.
    torch / ultralytics 불필요.
    필요: tensorrt, pycuda, numpy, opencv.
    """

    def __init__(self, model_path=None, conf=0.45, imgsz=640, iou=0.45):
        self.model_path = None
        self.engine = None
        self.context = None
        self.runtime = None
        self.trt = None
        self.cuda = None
        self.stream = None
        self.input_name = None
        self.output_names = []
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.bindings = {}
        # 재사용할 GPU 버퍼 (프레임마다 새로 잡지 않는다)
        self._buf_key = None
        self._dev_in = None
        self._host_out = []
        self._dev_out = []
        self._canvas = None       # letterbox 캔버스 재사용
        self._canvas_box = None
        self._ctx = None          # 이 객체를 만든 스레드가 소유하는 CUDA 컨텍스트
        if model_path:
            self.load(model_path)

    def _imports(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        self.trt = trt
        self.cuda = cuda

        # pycuda.autoinit 을 쓰지 않는다.
        # autoinit 은 컨텍스트를 "import 한 스레드"에 만들고, 프로그램이 끝날 때
        # atexit 으로 pop 한다. 추론을 전담 스레드에서 돌리면 만든 스레드와
        # 정리하는 스레드가 달라져서 종료할 때 아래 오류가 난다.
        #   context::pop failed: invalid device context
        # 여기서 직접 만들고 close() 에서 같은 스레드가 정리한다.
        # 컨텍스트는 프로세스에 하나만 둔다.
        #
        # 앙상블로 디텍터를 여러 개 띄울 때 각자 make_context() 를 부르면
        # 새 컨텍스트가 스택에 쌓여 앞엣것이 현재 컨텍스트가 아니게 된다.
        # 그러면 먼저 만든 스트림·버퍼가 무효가 되어 추론이 이렇게 죽는다.
        #   Cuda Runtime (invalid resource handle)
        # 그래서 하나를 만들어 공유하고, 마지막 사용자가 정리한다.
        global _CUDA_CTX, _CUDA_USERS
        if self._ctx is None:
            if _CUDA_CTX is None:
                cuda.init()
                _CUDA_CTX = cuda.Device(0).make_context()
            self._ctx = _CUDA_CTX
            _CUDA_USERS += 1

    def load(self, model_path):
        self._imports()

        p = Path(model_path).expanduser().resolve()
        if p.suffix.lower() != ".engine":
            raise ValueError("POS 추론 모델은 Jetson에서 생성한 .engine 파일이어야 합니다.")
        if not p.exists():
            raise FileNotFoundError(str(p))

        logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(logger)

        with open(p, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("TensorRT ENGINE 역직렬화에 실패했습니다.")

        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.Stream()
        self._buf_key = None      # 엔진이 바뀌었으니 버퍼를 다시 잡는다

        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("ENGINE 입력 텐서를 찾을 수 없습니다.")

        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if any(v < 0 for v in shape):
            # Ultralytics ONNX 기본 입력: NCHW
            self.context.set_input_shape(self.input_name, (1, 3, self.imgsz, self.imgsz))
        elif len(shape) == 4 and shape[2] > 0 and shape[3] > 0:
            # 입력이 고정된 엔진이면 그 크기를 따라가야 한다.
            #
            # imgsz=416 으로 내보낸 ONNX 를 640 으로 전처리해 넣으면 TensorRT 는
            # 버퍼 앞부분(416x416x3 float)만 읽어간다. NCHW 라서 그건 R 채널
            # 전부와 G 채널 일부를 이어붙인 것 — 사람이 보면 잡음이다.
            # 그래서 "검출이 아예 없거나 엉뚱한 자리에 뜨는" 증상이 난다.
            engine_size = int(shape[2])
            if engine_size != self.imgsz:
                print(f"[모델] 엔진 입력이 {shape[2]}x{shape[3]} 입니다. "
                      f"imgsz 를 {self.imgsz} → {engine_size} 로 맞춥니다.")
                self.imgsz = engine_size
                self._canvas = None          # 캔버스를 새 크기로 다시 만든다
                self._canvas_box = None
                self._buf_key = None         # GPU 버퍼도 다시 잡는다

        self.model_path = str(p)
        return True

    def ready(self):
        return self.engine is not None and self.context is not None

    def _letterbox(self, image, size=640):
        """비율을 유지한 채 size x size 캔버스에 얹는다.

        캔버스를 매번 np.full 로 새로 만들면 640x640x3 을 프레임마다 채우게 된다.
        크기가 늘 같으므로 한 장 만들어 두고 여백만 다시 칠한다.
        """
        h, w = image.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        left = (size - nw) // 2
        top = (size - nh) // 2

        canvas = self._canvas
        if canvas is None or canvas.shape[0] != size:
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            self._canvas = canvas
            self._canvas_box = None
        box = (left, top, nw, nh)
        if self._canvas_box != box:
            canvas[:] = 114          # 이전보다 작아졌을 때 옛 그림이 남지 않게
            self._canvas_box = box
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def _prepare(self, frame):
        img, ratio, pad_x, pad_y = self._letterbox(frame, self.imgsz)
        # blobFromImage 한 번이 cvtColor + transpose + astype + /255 를 모두 한다.
        # OpenCV 안에서 병렬로 도는 C++ 구현이라 numpy 로 따로 하는 것보다 빠르다.
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, swapRB=True)
        return blob, ratio, pad_x, pad_y

    def _alloc(self, inp):
        """입출력 GPU 버퍼를 한 번만 잡아 두고 계속 재사용한다.

        전에는 추론할 때마다 cuda.mem_alloc 을 새로 했다. 프레임마다 GPU 메모리를
        할당·해제하면 그만큼 시간이 들고, 해제가 파이썬 GC 에 맡겨져 있어
        메모리 사용량도 들쭉날쭉해진다. 입력 크기는 항상 같으므로 한 번이면 된다.
        """
        key = (inp.shape, inp.dtype.str)
        if self._buf_key == key:
            return
        self._host_in = np.ascontiguousarray(inp)
        self._dev_in = self.cuda.mem_alloc(self._host_in.nbytes)
        self._host_out = []
        self._dev_out = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(v < 0 for v in shape):
                shape = tuple(self.engine.get_tensor_shape(name))
            if any(v < 0 for v in shape):
                raise RuntimeError(f"동적 출력 shape 해석 실패: {name} {shape}")
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            self._host_out.append(host)
            self._dev_out.append(self.cuda.mem_alloc(host.nbytes))
        self._buf_key = key

    def _execute(self, inp):
        # TRT 10 tensor-address API
        in_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if any(v < 0 for v in in_shape):
            self.context.set_input_shape(self.input_name, inp.shape)

        input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        inp = np.ascontiguousarray(inp.astype(input_dtype, copy=False))

        self._alloc(inp)
        self.cuda.memcpy_htod_async(self._dev_in, inp, self.stream)
        self.context.set_tensor_address(self.input_name, int(self._dev_in))
        for name, dev in zip(self.output_names, self._dev_out):
            self.context.set_tensor_address(name, int(dev))

        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 실패")

        for host, dev in zip(self._host_out, self._dev_out):
            self.cuda.memcpy_dtoh_async(host, dev, self.stream)

        self.stream.synchronize()
        return self._host_out

    def _decode(self, outputs, ratio, pad_x, pad_y, original_shape):
        if not outputs:
            return []

        # 일반 Ultralytics detect ONNX는 첫 출력 하나 사용
        out = np.asarray(outputs[0])
        out = np.squeeze(out)

        # NMS 포함 export의 [N,6] 또는 [1,N,6]
        if out.ndim == 2 and out.shape[-1] == 6:
            rows = out
            boxes, scores, class_ids = [], [], []
            for row in rows:
                x1, y1, x2, y2, score, cls = row[:6]
                if float(score) < self.conf:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(score))
                class_ids.append(int(cls))
            return self._scale_detections(boxes, scores, class_ids, ratio, pad_x, pad_y, original_shape)

        # raw YOLO output: [C,N] or [N,C]
        if out.ndim != 2:
            raise RuntimeError(f"지원하지 않는 YOLO 출력 shape: {out.shape}")

        if out.shape[0] < out.shape[1] and out.shape[0] <= 512:
            out = out.T

        if out.shape[1] < 5:
            raise RuntimeError(f"지원하지 않는 YOLO 출력 shape: {out.shape}")

        xywh = out[:, :4]
        class_scores = out[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        mask = scores >= self.conf
        xywh = xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(xywh) == 0:
            return []

        # xywh center -> xyxy
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        nms_boxes = []
        for b in xyxy:
            nms_boxes.append([
                float(b[0]), float(b[1]),
                float(b[2] - b[0]), float(b[3] - b[1])
            ])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.astype(float).tolist(),
            self.conf,
            self.iou
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).reshape(-1)
        indices = self._drop_nested(xyxy, scores, indices)
        boxes = xyxy[indices].tolist()
        scores = scores[indices].astype(float).tolist()
        class_ids = class_ids[indices].astype(int).tolist()

        return self._scale_detections(boxes, scores, class_ids, ratio, pad_x, pad_y, original_shape)

    # 작은 박스가 큰 박스 안에 이만큼 들어가면 같은 물체를 두 번 잡은 것으로 본다
    NESTED_RATIO = 0.85

    @classmethod
    def _drop_nested(cls, xyxy, scores, indices):
        """큰 박스 안에 통째로 들어간 작은 박스 쌍에서 신뢰도가 낮은 쪽을 버린다.

        NMS 는 IoU(겹친 넓이 / 합친 넓이) 로 거른다. 그런데 와플 하나를
        작은 "waffle" 박스와 트레이를 덮는 큰 "pizza" 박스로 동시에 잡으면,
        작은 게 큰 것 안에 완전히 들어가도 IoU 는 0.16 정도라 0.45 문턱을
        못 넘어 둘 다 남는다. 화면에 박스가 겹쳐 뜨고 큰 쪽 이름이 엉뚱한 증상.

        그래서 IoS(겹친 넓이 / 작은 쪽 넓이) 를 본다. 작은 박스가 85% 이상
        안에 들어가면 같은 물체로 보고, 둘 중 신뢰도가 낮은 것을 뺀다.
        빵끼리 서로 속에 들어가 있는 일은 트레이 위에서 거의 없다.
        """
        if len(indices) < 2:
            return indices
        order = sorted(indices, key=lambda i: -float(scores[i]))
        keep = []
        for i in order:
            a = xyxy[i]
            aa = max(1e-6, float((a[2] - a[0]) * (a[3] - a[1])))
            nested = False
            for j in keep:
                b = xyxy[j]
                bb = max(1e-6, float((b[2] - b[0]) * (b[3] - b[1])))
                iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                if iw * ih / min(aa, bb) >= cls.NESTED_RATIO:
                    nested = True          # 이미 남긴(더 확신 있는) 박스와 같은 물체
                    break
            if not nested:
                keep.append(i)
        return np.array(keep, dtype=int)

    @staticmethod
    def _scale_detections(boxes, scores, class_ids, ratio, pad_x, pad_y, original_shape):
        h, w = original_shape[:2]
        dets = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_x) / ratio
            y1 = (y1 - pad_y) / ratio
            x2 = (x2 - pad_x) / ratio
            y2 = (y2 - pad_y) / ratio

            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))

            dets.append({
                "class_id": int(cls_id),
                "confidence": float(score),
                "xyxy": [x1, y1, x2, y2],
            })
        return dets

    def close(self):
        """CUDA 컨텍스트를 정리한다. 반드시 만든 스레드에서 불러야 한다."""
        if self._ctx is None:
            return
        self._buf_key = None
        self._dev_in = None
        self._dev_out = []
        self.context = None
        self.engine = None
        global _CUDA_CTX, _CUDA_USERS
        _CUDA_USERS = max(0, _CUDA_USERS - 1)
        if _CUDA_USERS == 0:
            try:
                self._ctx.pop()
                self._ctx.detach()
            except Exception:
                pass
            _CUDA_CTX = None
        self._ctx = None

    def _detect(self, frame):
        """한 장을 한 번 본다. 결과는 파이썬 목록으로 바로 풀어 둔다.

        _execute 는 GPU 출력 버퍼를 재사용하므로, 다음 추론 전에
        여기서 반드시 _decode 까지 끝내야 결과가 덮어쓰이지 않는다.
        """
        inp, ratio, pad_x, pad_y = self._prepare(frame)
        outputs = self._execute(inp)
        return self._decode(outputs, ratio, pad_x, pad_y, frame.shape)

    def infer(self, frame, draw=True, tta=False):
        """(그려진 프레임, 검출목록, 클래스별 개수).

        draw=False 면 박스를 그리지 않고 원본을 그대로 돌려준다.
        실시간 미리보기는 화면에 맞춘 뒤 따로 그리므로 여기서 그릴 필요가 없다.

        tta=True 면 좌우반전으로 한 번 더 보고 합친다 (시간 약 2배: 19 → 39ms).
        실시간 미리보기(33ms 주기)에는 못 쓰고, 「빵 인식」 버튼처럼 한 번만
        볼 때 쓴다. 실측(휘낭시에, 20프레임): 원본 방향에서 rotus 0.22 로
        잘못 약하게 잡히던 빵이 반전하면 fruit 0.76 으로 잡혀
        POS 통과 0/20 → 20/20. 와플·plain 등 다른 빵은 그대로 20/20.

        합칠 때 cross_class 를 켠다. 원본과 반전이 같은 빵을 다른 이름으로
        부르면 둘 다 남아 박스가 겹쳐 뜨기 때문이다 (끄면 20프레임에 20건 겹침).
        """
        if not self.ready():
            raise RuntimeError("활성 TensorRT .engine 모델이 없습니다.")

        detections = self._detect(frame)
        if tta:
            from ai.fuse import fuse
            w = frame.shape[1]
            flipped = self._detect(cv2.flip(frame, 1))
            flipped = [{**d, "xyxy": [w - d["xyxy"][2], d["xyxy"][1],
                                      w - d["xyxy"][0], d["xyxy"][3]]}
                       for d in flipped]
            detections = fuse([detections, flipped], mode="agree", cross_class=True)

        if not draw:
            counts = {}
            for d in detections:
                counts[d["class_id"]] = counts.get(d["class_id"], 0) + 1
            return frame, detections, counts

        # 그리기는 원격 추론과 공용 (ai/render.py)
        plotted, counts = draw_detections(frame, detections)
        return plotted, detections, counts
