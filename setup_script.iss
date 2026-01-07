; ==========================================================
; HAIA Installer Script (Hybrid Structure)
; ==========================================================
#define MyAppName "HAIA"
#define MyAppVersion "0.26"
#define MyAppPublisher "My Name"
#define MyAppExeName "haia.exe"

; ソースパスの定義（環境に合わせて調整してください）
; Nuitkaのビルド出力フォルダ
#define BuildSourceDir "C:\projects\HAIA\haia.dist"
; 設定ファイルやドキュメントがあるプロジェクトルート
#define ProjectRoot "C:\projects\HAIA"

[Setup]
; --- 基本情報 ---
AppId={{B2405A9D-9D10-4ACD-9FE5-EF5AC5D3F8F0}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; --- 権限設定 (ハイブリッドモード: 元のスクリプトを維持) ---
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; --- インストール先の設定 (Codeセクションで動的に決定) ---
DefaultDirName={code:GetDefaultDir}

; --- その他の設定 ---
DisableDirPage=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DirExistsWarning=no
OutputBaseFilename=HAIA_v{#MyAppVersion}_Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
; インストール先ルートの権限設定（管理者インストールの場合はUsersに書き込み権限付与）
Name: "{app}"; Permissions: users-modify; Check: IsAdminInstallMode

[Files]
; ----------------------------------------------------------
; 1. プログラム本体群 (binフォルダへ隔離)
; ----------------------------------------------------------
; Nuitkaの出力フォルダの中身を、インストール先の {app}\bin に入れます
; 元のスクリプトの参照先を使用 [cite: 4]
Source: "{#BuildSourceDir}\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs

; ----------------------------------------------------------
; 2. ルートに置く設定ファイル・ランチャー
; ----------------------------------------------------------
; ※これらのファイルが ProjectRoot にあると仮定しています。パスは適宜調整してください。
Source: "{#ProjectRoot}\launch_haia.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\tasks.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\models.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\settings.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\casting_config.json"; DestDir: "{app}"; Flags: ignoreversion

; ----------------------------------------------------------
; 3. ドキュメント・ライセンス関連
; ----------------------------------------------------------
Source: "{#ProjectRoot}\readme.html"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\readme.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\readme.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\DISCLAIMER.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\企業内利用向け注意事項.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\CORPORATE_USE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\users_manual.pdf"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}"; Flags: ignoreversion

; ----------------------------------------------------------
; 4. プロンプトフォルダ
; ----------------------------------------------------------
Source: "{#ProjectRoot}\prompts\*"; DestDir: "{app}\prompts"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; スタートメニュー
; ★重要★ WorkingDir: "{app}" を指定することで、アプリはbinの親フォルダ(ルート)を作業場所として認識します
Name: "{group}\{#MyAppName}"; Filename: "{app}\bin\{#MyAppExeName}"; WorkingDir: "{app}"

; デスクトップ (ユーザー別 or 全ユーザー自動切替)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\bin\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
; インストール完了後の実行
; 実行ファイルは bin の中にあるためパスを変更しています
Filename: "{app}\bin\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

; ドキュメント表示など（オプション）
Filename: "{app}\readme.html"; Description: "Readmeを表示する"; Flags: shellexec nowait postinstall runasoriginaluser skipifsilent unchecked

[Code]
// --- インストール先を動的に決定する関数 (元のスクリプトを維持 ) ---
function GetDefaultDir(Param: String): String;
begin
  if IsAdminInstallMode then
  begin
    // 管理者モード: C:\HAIA ({sd}\HAIA)
    Result := ExpandConstant('{sd}\{#MyAppName}');
  end
  else
  begin
    // ユーザーモード: %LOCALAPPDATA%\HAIA
    Result := ExpandConstant('{localappdata}\{#MyAppName}');
  end;
end;
