<!--
  연구실 README 템플릿 (research code release)

  사용법
  1. 이 파일을 레포 루트에 README.md 로 복사합니다.
  2. 대괄호 [ ]를 실제 값으로 채웁니다.
  3. 해당 없는 섹션은 통째로 삭제합니다. (빈 섹션을 남겨두는 것이 제일 나쁩니다)
  4. 이 주석들은 안내용이므로 전부 지웁니다.
  5. "INTERNAL" 블록은 연구실 내부용이며, public 전환 시 반드시 삭제합니다.

  원칙
  - 독자는 "이 코드를 처음 보는 외부 연구자"입니다. 우리 서버 사정, 우리 관례를
    모른다고 가정하고 쓰세요.
  - 명령어는 복사-붙여넣기해서 그대로 돌아가야 합니다. 경로/파일명은 문서 전체에서
    한 글자도 다르지 않게 통일하세요.
-->

# [ProjectName]: [Full Paper Title]

**[Venue] [Year]**

[Author One](URL)<sup>1</sup>, [Author Two](URL)<sup>1</sup>, [Author Three](URL)<sup>2</sup>
<sup>1</sup>[Affiliation 1], <sup>2</sup>[Affiliation 2]

[**Paper**](URL) | [**arXiv**](URL) | [**Project Page**](URL) | [**Video**](URL) | [**Poster**](URL)

<!-- - Paper 링크는 학회 홈페이지(페이월/미공개)보다 arXiv 또는 PDF 직링크를 우선합니다.
     아직 없으면 arXiv 업로드 후 교체하세요.
     - 없는 링크는 지웁니다. 죽은 링크를 남기면 레포 전체의 신뢰도가 떨어집니다.
     - 저자/소속 줄이 부담스러우면 통째로 지워도 됩니다. -->

---

<!-- ===== INTERNAL — public 전환 시 이 블록 전체 삭제 ===== -->
> **담당자** @github-id · **상태** [진행 중 / 논문 투고 중 / 공개 준비 완료] · **최종 확인** [YYYY-MM-DD]
>
> **내부 메모**
> - 코드/데이터/체크포인트 서버 경로: `[/data/...]`, `[/nas/...]`
> - 재현 환경: `[서버명]`, `[conda env 이름]`
> - 삽질 기록: [예: batch 32 이상에서 OOM → gradient accumulation 사용]
> - 알려진 이슈: [예: multi-GPU 학습 시 결과 재현 안 됨]
> - 인수인계 시 확인할 것: [ ]
<!-- ===== INTERNAL 끝 ===== -->

---

## Introduction

This is the official implementation of **[Full Paper Title]** ([Venue] [Year]).

[2~4문장으로 요약합니다. 최소한 다음 세 가지가 들어가야 합니다.
 (1) 무엇을 입력받아 무엇을 출력하는가
 (2) 핵심 아이디어 한 줄
 (3) 기존 방법 대비 무엇이 나아졌는가 (가능하면 숫자로)]

For more details, please refer to our [paper](URL) and [project page](URL).

<p align="center">
  <img src="assets/teaser.png" width="80%">
</p>

<!-- teaser 이미지나 gif 하나면 체감이 크게 달라집니다.
     assets/ 폴더에 두고 상대 경로로 거세요. 외부 이미지 호스팅은 언젠가 죽습니다.
     용량은 gif 기준 10MB 이하를 권장합니다 (git 히스토리에 영구히 남습니다). -->

### News

- **[YYYY-MM-DD]** Code and pre-trained weights released.
- **[YYYY-MM-DD]** Paper accepted to [Venue] [Year].

<!-- News는 업데이트할 자신이 있을 때만 넣으세요. 1년째 멈춰 있으면 방치된 레포로 보입니다. -->

## Table of Contents

- [\[ProjectName\]: \[Full Paper Title\]](#projectname-full-paper-title)
  - [Introduction](#introduction)
    - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
    - [Optional dependencies](#optional-dependencies)
  - [Repository Structure](#repository-structure)
  - [Data Preparation](#data-preparation)
    - [1. Download](#1-download)
    - [2. Preprocess](#2-preprocess)
    - [3. Expected layout](#3-expected-layout)
  - [Pre-trained Models](#pre-trained-models)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference on your own data](#inference-on-your-own-data)
  - [Reproducing Paper Results](#reproducing-paper-results)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)

<!-- 반드시 상대 앵커(#installation)를 쓰세요.
     절대 URL을 쓰면 fork되거나 레포 이름이 바뀌는 순간 전부 깨집니다.
     앵커는 헤딩 텍스트를 소문자로 바꾸고 공백을 '-'로 치환한 형태입니다. -->

## Installation

### Requirements

- [Ubuntu 20.04 / 22.04]
- [Python 3.10]
- [CUDA 11.8] (GPU 사용 시)
- [PyTorch 2.1]

Tested on [NVIDIA RTX 4090, 24GB] with [Ubuntu 22.04], [CUDA 11.8].
If your environment differs, you may need to adjust the [PyTorch/CUDA] versions accordingly.

<!-- "우리 서버에서는 됐다"가 아니라 "무엇 위에서 검증했다"를 적는 것이 핵심입니다.
     GPU 메모리 요구량을 함께 적어주면 문의가 확 줄어듭니다. -->

### Setup

```bash
git clone https://github.com/[org]/[repo-name].git
cd [repo-name]
```

<!-- 아래 세 가지 중 실제로 쓰는 방식 하나만 남기고 나머지는 삭제하세요.
     여러 방법을 나열하면 어느 것도 검증되지 않은 상태가 되기 쉽습니다. -->

**Option A — conda**

```bash
conda env create -f environment.yml
conda activate [env-name]
```

**Option B — pip (venv)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Option C — Docker**

```bash
docker build -t [image-name] .
docker run --gpus all -it -v $(pwd):/workspace [image-name]
```

> [!IMPORTANT]
> All commands below assume the `[env-name]` environment is active and that you are
> in the repository root.

<!-- GitHub은 > [!NOTE] > [!TIP] > [!IMPORTANT] > [!WARNING] > [!CAUTION] 를
     컬러 박스로 렌더링합니다. 대괄호와 느낌표 형식을 정확히 지켜야 하고,
     반드시 인용문(>) 안에 있어야 합니다. -->

### Optional dependencies

[특정 기능에만 필요한 무거운 의존성(예: 시각화, 특정 CUDA 확장)은 여기서 분리합니다.]

```bash
pip install -r requirements-[optional].txt
```

## Repository Structure

```
[repo-name]/
├── configs/            # 실험 설정 파일
├── data/               # 데이터셋 (git에 포함되지 않음)
├── scripts/            # 논문 결과 재현용 실행 스크립트
├── src/[package]/
│   ├── models/         # 모델 정의
│   ├── datasets/       # 데이터 로더
│   └── ...
├── tools/              # train / eval / inference 진입점
├── environment.yml
└── README.md
```

<!-- 이 섹션은 선택 사항이지만, 파일이 많은 레포라면 넣는 편이 훨씬 낫습니다.
     단, 실제 구조와 어긋나는 순간 없느니만 못합니다. 구조가 바뀌면 같이 고치세요.
     디렉토리 이름은 본문·명령어·트리에서 항상 동일하게 쓰세요.
     data / datasets 가 섞이면 그대로 이슈로 돌아옵니다. -->

## Data Preparation

### 1. Download

| Dataset | Description | Size | Link |
|---|---|---|---|
| [Dataset A] | [설명] | [XX GB] | [Download](URL) |
| [Dataset B] | [설명] | [XX GB] | [Official page](URL) |

> [!NOTE]
> [일부 데이터셋은 저자에게 별도 신청이 필요합니다 / 라이선스 동의가 필요합니다.]

### 2. Preprocess

```bash
python tools/prepare_data.py --dataset [dataset-name] --root [data/[dataset-name]]
```

### 3. Expected layout

After preprocessing, the data directory should look like this:

```
data/
└── [dataset-name]/
    ├── train/
    ├── val/
    └── [annotations.json]
```

<!-- "어떤 상태가 되면 성공인지"를 보여주는 트리가 가장 중요합니다.
     사용자는 이걸로 자기 상태를 검증합니다.
     데이터 경로를 코드에 하드코딩하지 말고 config나 인자로 받게 하세요. -->

## Pre-trained Models

We provide pre-trained weights so that [the training stage can be skipped].

| Model | Dataset | [Metric] | Download |
|---|---|---|---|
| [model-a] | [Dataset A] | [XX.X] | [ckpt](URL) |
| [model-b] | [Dataset B] | [XX.X] | [ckpt](URL) |

Place the downloaded files as follows:

```
[repo-name]/
└── checkpoints/
    ├── [model_a.pth]
    └── [model_b.pth]
```

<!-- ★ 파일 이름은 이 트리와 아래 Usage 명령어에서 정확히 일치시키세요.
     트리에는 model_a.pth, 명령어에는 modelA.pt 라고 써 두면
     사용자는 어느 쪽이 맞는지 알 수 없습니다. 이건 가장 흔한 이슈 유형입니다.
     체크포인트는 Google Drive보다 Hugging Face Hub / Zenodo 를 권장합니다.
     (용량 제한이 없고, 링크가 오래 살아남고, DOI를 받을 수 있습니다) -->

## Usage

<!-- 아래는 일반적인 3단계 구성입니다. 프로젝트가 다단계 파이프라인이라면
     "### Stage 1 — [이름]" 형태로 바꾸고, 각 단계의 입력/출력/소요 시간을 적으세요. -->

### Training

Required inputs:

- [Dataset] (prepared as described in [Data Preparation](#data-preparation))
- [Config file] (`configs/[name].yaml`) — [무엇을 바꾸면 되는지 한 줄]

```bash
python tools/train.py \
  --config configs/[name].yaml \
  --data-root data/[dataset-name] \
  --output-dir outputs/[exp-name]
```

Checkpoints and logs are written to `outputs/[exp-name]/`.
Training takes approximately **[N] hours on [1× RTX 4090]** and requires about **[XX] GB** of GPU memory.

Multi-GPU training:

```bash
torchrun --nproc_per_node=[N] tools/train.py --config configs/[name].yaml
```

### Evaluation

```bash
python tools/eval.py \
  --config configs/[name].yaml \
  --checkpoint checkpoints/[model_a.pth] \
  --split [test]
```

Expected output:

```
[Metric1]: [XX.X]
[Metric2]: [XX.X]
```

### Inference on your own data

```bash
python tools/inference.py \
  --checkpoint checkpoints/[model_a.pth] \
  --input [path/to/input] \
  --output [outputs/demo]
```

[입력 형식 요구사항을 한두 줄로: 예) 이미지는 RGB, 최소 해상도 XXX, intrinsic 필요 여부]

<!-- ★ 각 명령어마다 반드시 적어야 하는 것 세 가지
     (1) 결과물이 어디에 생기는가
     (2) 대략 얼마나 걸리는가
     (3) 자원이 얼마나 필요한가 (GPU 메모리 등)
     이게 없으면 "멈춘 건가 도는 건가" 문의가 반드시 옵니다. -->

## Reproducing Paper Results

| Paper | Command | Expected output |
|---|---|---|
| Table 1 | `bash scripts/table1.sh` | [Accuracy 87.3] |
| Table 2 (ablation) | `bash scripts/table2.sh` | `outputs/ablation.csv` |
| Figure 4 | `python scripts/fig4.py` | `outputs/fig4.pdf` |

[시드 고정 여부와 결과가 흔들리는 범위를 한 줄로 적으세요.
 예: "Seeds are fixed (seed=0). Results may vary by ±0.3 depending on GPU/CUDA version."]

<!-- 논문 표와 명령어를 1:1로 묶어두는 섹션입니다.
     리비전 대응, 리뷰어 재현 요청, 그리고 6개월 뒤의 자기 자신에게 전부 필요합니다.
     스크립트를 실제로 clean clone에서 한 번 돌려보고 채우세요. -->

## Troubleshooting

<details>
<summary><b>[CUDA out of memory]</b></summary>

[해결 방법: 예) `--batch-size`를 줄이거나 `--grad-accum`을 사용하세요.]

</details>

<details>
<summary><b>[특정 패키지 설치 실패]</b></summary>

[해결 방법]

</details>

<!-- 랩 내부에서 이미 겪은 문제를 여기 적어두면 이슈의 절반이 사라집니다.
     이슈로 들어온 질문은 답변 후 여기에 반영하세요.
     문제가 없다면 이 섹션은 삭제합니다. -->

## License

This project is released under the [LICENSE-NAME] license. See [LICENSE](LICENSE) for details.

> [!NOTE]
> This code builds on [[BaseWork]](URL), which is distributed under [its license].
> Please review the licenses of all upstream works and datasets before use.

<!-- ★★ 공개 전 반드시 확인 ★★
     - 상당수의 연구 코드/데이터셋은 non-commercial research only 라이선스입니다.
       그 위에 쌓아 올린 코드에는 상위 라이선스 제약이 그대로 따라옵니다.
     - 업스트림 라이선스를 확인하지 않고 MIT로 공개하면 라이선스 위반이 될 수 있습니다.
     - LICENSE 파일이 없는 공개 레포는 법적으로 "아무 권리도 주지 않음"입니다.
       반드시 LICENSE 파일을 포함하세요.
     - 데이터셋 재배포 가능 여부도 별도로 확인하세요. 대부분 원본 링크만 걸어야 합니다. -->

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{[key],
    title     = {[Full Paper Title]},
    author    = {[Author One and Author Two and Author Three]},
    booktitle = {[Proceedings of ...]},
    year      = {[2026]}
}
```

## Acknowledgements

This work builds upon the following projects. We thank the authors for releasing their code:

- [[Work1]](URL)
- [[Work2]](URL)

[연구비 지원 문구가 필요하면 여기에 넣으세요.]

## Contact

For questions, please open an [issue](https://github.com/[org]/[repo-name]/issues)
or contact [name] ([email]).

<!-- ===== INTERNAL — 공개 전 체크리스트. public 전환 시 삭제 =====

  [ ] INTERNAL 블록과 이 체크리스트를 모두 삭제했다
  [ ] 모든 대괄호 [ ]를 실제 값으로 채웠다 (Ctrl+F로 '[' 검색)
  [ ] 안내 주석을 전부 지웠다
  [ ] LICENSE 파일을 추가했고, 업스트림 라이선스와 충돌하지 않는다
  [ ] .gitignore가 적용되어 있고 데이터/체크포인트가 커밋되지 않았다
  [ ] 커밋 히스토리에 키/토큰/비밀번호/서버 경로가 남아 있지 않다
      (git log -p | grep -iE 'api[_-]?key|token|password|secret' 로 확인)
  [ ] 절대 경로(/home/..., /data/...)가 코드와 config에 남아 있지 않다
  [ ] 새 환경에서 clean clone → 설치 → 추론까지 실제로 한 번 성공했다
  [ ] README의 모든 링크가 살아 있다
  [ ] 체크포인트 다운로드 링크가 로그인 없이 접근 가능하다
  [ ] 공동저자/지도교수 승인을 받았다

===== INTERNAL 끝 ===== -->
