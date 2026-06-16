# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Vulkanを学習するためのC++プロジェクト。GLFW + Vulkan APIを使って3Dモデルをレンダリングする。

**重要な方針：** ユーザーはVulkanを学習中のため、Vulkan APIに関わる実装（パイプライン作成、デスクリプタ設定、コマンドバッファ記録など）は自分で書けるよう、実装の意図・選択肢・トレードオフを説明し、コードの枠組みを用意した上でユーザーに書いてもらうこと。ボイラープレートや設定コードは代わりに実装してよい。

**応答言語：** 必ず日本語で応答する。

## ビルド

Visual Studio で `Vulkan.sln` を開いてビルド。CLIビルドは以下：

```
msbuild Vulkan.sln /p:Configuration=Release /p:Platform=x64
```

シェーダのコンパイル（`Shaders/` ディレクトリ内で実行）：

```
ShaderCompile.bat
```

glslcのパス: `F:/work/library/VulkanSDK/1.4.321.1/Bin/glslc.exe`

## アーキテクチャ

### 初期化フロー（VulkanApp.cpp）

```
App::run()
  └─ VulkanApp コンストラクタ
       ├─ VulkanContext::InitializeVulkan()  // Instance, PhysicalDevice, LogicalDevice
       ├─ VulkanSwapChain::Initialize()      // SwapChain, ImageView, Depth, Framebuffer
       ├─ createRenderPass()
       ├─ ShaderCPUResource()                // SPIRV-Cross反射 → UBO/Texture/DescriptorSet
       ├─ createGraphicsPipeline()
       └─ モデル・テクスチャの読み込み
```

### 主要クラスの責務

| クラス | 役割 |
|--------|------|
| `VulkanContext` | Instance・PhysicalDevice・LogicalDevice・Queue・ValidationLayerの初期化 |
| `VulkanSwapChain` | SwapChain・ImageView・DepthBuffer・Framebufferの管理、リサイズ対応 |
| `VulkanCommandBuffer` | CommandPool・CommandBufferの生成と管理 |
| `VulkanResources` | ShaderModule・Image・ImageView・DescriptorSetの生成ユーティリティ |
| `VulkanBufferCreator` | Vertex/Index/UniformBufferの確保とステージングバッファ転送 |
| `VulkanTextureCreator` | テクスチャ画像の読み込み・VkImage生成・Sampler作成 |
| `ShaderCPUResources` | SPIRV-CrossでSPV反射 → DescriptorSetLayout・UBO・DescriptorSetを自動構築 |
| `ShaderPropertyApplier` | マップ済みUBOメモリへのカメラ・モデル行列書き込み |
| `IModelLoader` | モデルローダ抽象インターフェース（Assimp版とtinyobjloader版が存在） |

### ディスクリプタセット構成（set番号）

- `set=0`: カメラUBO（view・proj行列）
- `set=1`: モデルUBO（model行列・color）+ テクスチャサンプラー

### シェーダ

- `Shaders/shader.vert` / `shader.frag` → `glslc` で → `vert.spv` / `frag.spv`
- 実行時にSPIRV-Crossでリフレクションし、バッファサイズ・バインディング・メンバーオフセットを自動取得

### フレームループ

`VulkanApp::Draw(currentFrame)` が毎フレーム呼ばれ、フェンス待機 → イメージ取得 → コマンドバッファ記録 → Submit → Present を行う。`MAX_FRAMES_IN_FLIGHT = 2` のダブルバッファリング。

## 主要な外部ライブラリ

- **Vulkan SDK 1.4.321.1**: `F:/work/library/VulkanSDK/`
- **GLFW**: ウィンドウ・入力管理
- **GLM**: 数学ライブラリ（`GLM_FORCE_DEPTH_ZERO_TO_ONE` を設定済み）
- **assimp**: 多フォーマットモデルローダ（`Library/assimp/`）
- **tinyobjloader**: OBJ特化の軽量ローダ
- **SPIRV-Cross**: シェーダリフレクション（`Library/SPIRV-Cross/`）

## アセット

- `Assets/viking_room.obj` + `Assets/viking_room.png` をデフォルトで読み込み
