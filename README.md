# SpikeAI — Nivelamento de Pose Estimation

Primeira etapa da pipeline de visão computacional do SpikeAI: transformar um vídeo de ataque em um novo vídeo contendo **somente o skeleton** sobre fundo vazio.

Não há nesta etapa análise biomecânica, detecção da bola, classificação do ataque nem feedback técnico.

## Pipeline

```text
Vídeo → OpenCV → Frames → MediaPipe Pose Landmarker → Landmarks → Renderer → Fundo vazio + skeleton → Vídeo de saída
```

## Tecnologias

- Python 3.12 (recomendado; MediaPipe 1.0 não suporta Python 3.14)
- OpenCV
- MediaPipe Pose Landmarker (Tasks API)
- Git

## Requisitos

- Windows, Linux ou macOS
- Python **3.12**
- Webcam não é necessária; a entrada é um arquivo `.mp4`

## Instalação

```bash
git clone <repository>
cd cortechx_spikeai

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Se o PowerShell bloquear `Activate.ps1`, execute com o interpretador do venv:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

## Execução

1. Coloque o vídeo do atleta em `input/ataque_volei.mp4`.
2. Rode:

```bash
python src/main.py
```

Na primeira execução o modelo `models/pose_landmarker_lite.task` é baixado automaticamente (Google Cloud Storage).

## Entrada e saída

| | Caminho |
|---|---|
| Entrada | `input/ataque_volei.mp4` |
| Saída (vídeo) | `output/ataque_volei_pose.mp4` |
| Saída (dados) | `output/landmarks.json` |

O vídeo de saída usa o FPS e a resolução do original. Frames sem pose viram fundo preto (o frame original **não** é usado como fallback).

Landmarks com `visibility` abaixo de 0.5 não são desenhados. O JSON guarda todos os landmarks do frame (mesmo com baixa visibility) e o ângulo do cotovelo esquerdo/direito.

Ao final da execução o terminal mostra estatísticas: frames com/sem pose, taxa de detecção e tempo total.

## Estrutura

```text
input/          vídeo de entrada (local; não versionado)
output/         vídeo e JSON gerados (não versionados)
models/         modelo .task do Pose Landmarker (baixado localmente)
src/main.py     orquestra o loop
src/video.py    leitura e escrita de vídeo
src/pose.py     MediaPipe + landmarks estruturados
src/renderer.py frame vazio + pontos + conexões
src/geometry.py visibility e ângulo do cotovelo
src/export.py   exportação JSON
requirements.txt
```

## Limitações

- Pose **2D** com profundidade relativa (`z`); não é reconstrução biomecânica 3D.
- Oclusão, baixa iluminação e movimentos muito rápidos reduzem a qualidade dos landmarks.
- Atleta parcialmente fora do quadro gera skeleton incompleto; fora por completo gera frame vazio.
- Com várias pessoas, o MVP usa **apenas a primeira pose** detectada.
- Coordenadas do MediaPipe são normalizadas `[0, 1]`; o renderer converte para pixels (`x * largura`, `y * altura`).
- No Windows o codec de escrita é `mp4v`.
- O modelo Lite prioriza velocidade em relação ao modelo Full.
