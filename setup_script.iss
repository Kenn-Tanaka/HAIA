; ==========================================================
; HAIA (Desktop AI Assistant) Installer Script
; ==========================================================

#define MyAppName "HAIA"
#define MyAppVersion "0.19"
#define MyAppPublisher "HAIA Project"
#define MyAppExeName "haia_v0_19.exe"
; ↑ Nuitkaが出力したEXE名

#define BuildSourceDir "haia.dist"
; ↑ Nuitkaが出力したフォルダ名

[Setup]
; --- 基本設定 ---
; AppIdはツールバーの [Tools] -> [Generate GUID] で必ず書き換えてください
AppId={{B2405A9D-9D10-4ACD-9FE5-EF5AC5D3F8F0}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; --- 出力設定 ---
OutputDir=.
OutputBaseFilename=HAIA_v0.19_Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

; --- インストール後の動作 ---
; 完了画面でReadmeを表示するオプション
DisableProgramGroupPage=yes

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; ----------------------------------------------------------
; 1. プログラム本体群 (binフォルダへ隔離)
; ----------------------------------------------------------
; Nuitkaの出力フォルダの中身を、インストール先の {app}\bin に入れます
Source: "{#BuildSourceDir}\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs

; ----------------------------------------------------------
; 2. ルートに置く設定ファイル・ランチャー
; ----------------------------------------------------------
Source: "launch_haia.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "tasks.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "models.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "settings.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "casting_config.json"; DestDir: "{app}"; Flags: ignoreversion

; ----------------------------------------------------------
; 3. ドキュメント・ライセンス関連
; ----------------------------------------------------------
Source: "readme.html"; DestDir: "{app}"; Flags: ignoreversion
;Source: "users_manual.pdf"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "THIRD_PARTY_LICENSES.txt"; DestDir: "{app}"; Flags: ignoreversion

; ----------------------------------------------------------
; 4. プロンプトフォルダ
; ----------------------------------------------------------
Source: "prompts\*"; DestDir: "{app}\prompts"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; スタートメニューのショートカット
; ターゲットは bin\haia.exe ですが、作業フォルダ(WorkingDir)をルート({app})にします
Name: "{group}\{#MyAppName}"; Filename: "{app}\bin\{#MyAppExeName}"; WorkingDir: "{app}"

; デスクトップショートカット
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\bin\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
; インストール完了後にReadmeを開く（ブラウザ起動）
; postinstallフラグにより、完了画面のチェックボックスとして表示されます
Filename: "{app}\readme.html"; Description: "導入ガイド (FFmpeg/Pandocの手順) を表示する"; Flags: shellexec nowait postinstall runasoriginaluser

; インストール完了後にアプリを起動する（オプション）
Filename: "{app}\bin\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent unchecked; WorkingDir: "{app}"