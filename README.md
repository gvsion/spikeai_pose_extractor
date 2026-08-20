# SpikeAI — Nivelamento de Pose Estimation

Primeira etapa da pipeline de visão computacional do SpikeAI: transformar um vídeo de ataque em um novo vídeo contendo **somente o skeleton** sobre fundo vazio.

Não há detecção da bola, classificação do ataque nem feedback técnico. O ângulo do cotovelo no vídeo/JSON é só uma demonstração de geometria (landmarks → vetores → graus), não o pipeline biomecânico do Spike.AI.

## Pipeline

```text
Vídeo → OpenCV → Frames → MediaPipe Pose Landmarker → Landmarks → Renderer → Fundo vazio + skeleton → Vídeo de saída
```

## Tecnologias

- Python **3.12**
- OpenCV
- MediaPipe Pose Landmarker (Tasks API)
- Git

## Requisitos

- Windows, Linux ou macOS
- Python 3.12
- Internet na **primeira execução** (download do arquivo `pose_landmarker_lite.task`)
- Um `.mp4` local (webcam não é usada neste projeto)
- Para assistir o MP4 gerado no Windows, use VLC se o player nativo não abrir (`mp4v`)

## Instalação

```bash
git clone git@github.com:gvsion/spikeai_pose_extractor.git
cd spikeai_pose_extractor

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Se o PowerShell bloquear `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

## Execução

1. Coloque o vídeo em `input/ataque_volei.mp4` (os `.mp4` não vão para o Git).
2. Rode:

```bash
python src/main.py
```

Na primeira execução o modelo é baixado para `models/pose_landmarker_lite.task`.

## Entrada e saída

| | Caminho |
|---|---|
| Entrada | `input/ataque_volei.mp4` |
| Somente skeleton (obrigatório) | `output/ataque_volei_pose.mp4` |
| Original + skeleton (validação) | `output/original_pose.mp4` |
| Dados | `output/landmarks.json` |

O vídeo obrigatório usa fundo preto: atleta, quadra, bola e rede **não** aparecem. Frames sem pose ficam pretos nesse arquivo.

`original_pose.mp4` mantém o vídeo original com o skeleton por cima, só para conferir a detecção.

Landmarks com `visibility` abaixo de 0.5 não são desenhados. O JSON guarda os pontos do frame com `x`/`y`/`z`/`visibility` arredondados a 4 casas e o ângulo do cotovelo (inteiro se fechado, 1 casa se quebrado). O ângulo também é escrito perto do cotovelo no vídeo.

Ao final, o terminal mostra frames com/sem pose, taxa de detecção e tempo total.

## Estrutura

```text
input/          coloque o .mp4 aqui (a pasta sobe no Git via .gitkeep; o vídeo não)
output/         criado na execução (vídeos e JSON; ignorado pelo Git)
models/         criado na execução (modelo .task baixado; ignorado)
src/main.py     orquestra o loop
src/video.py    leitura e escrita de vídeo
src/pose.py     MediaPipe + landmarks estruturados
src/renderer.py frame vazio + pontos + conexões + ângulo
src/geometry.py visibility e ângulo do cotovelo
src/export.py   exportação JSON
requirements.txt
```

## Limitações

- Pose **2D** com profundidade relativa (`z`); não é reconstrução biomecânica 3D.
- O ângulo do cotovelo é 2D no plano da imagem (em pixels), não o ângulo articular 3D.
- Oclusão, baixa iluminação e movimentos muito rápidos reduzem a qualidade dos landmarks.
- Atleta parcialmente fora do quadro gera skeleton incompleto; fora por completo gera frame vazio no vídeo obrigatório.
- Com várias pessoas, usa **apenas a primeira pose** detectada.
- Coordenadas do MediaPipe são normalizadas `[0, 1]`; o renderer converte para pixels (`x * largura`, `y * altura`).
- Codec de escrita: `mp4v` (VLC costuma abrir; Filmes e TV do Windows às vezes não).
- O modelo Lite prioriza velocidade em relação ao modelo Full.
