; Inno Setup script for the GliaAnalysis Windows installer.
; Build the app first (pyinstaller packaging\gliaanalysis.spec --noconfirm), then
; compile this script with Inno Setup (iscc packaging\windows\gliaanalysis.iss).
; Installs per user (no administrator rights needed).

#define AppName "GliaAnalysis"
; The release workflow passes the version (iscc /DAppVersion=x.y.z); otherwise
; keep this in step with glia_dash/__init__.py
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppExe "GliaAnalysis.exe"

[Setup]
AppId={{6310BE24-AAB4-43A5-835D-E7FAF4303068}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Epilepsy Center, Lund University
AppPublisherURL=https://github.com/EpilepsyCenter/GliaAnalysis
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=GliaAnalysis-{#AppVersion}-Setup
SetupIconFile=..\..\glia_dash\assets\glia_icon.ico
UninstallDisplayIcon={app}\{#AppExe}
LicenseFile=..\..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\GliaAnalysis\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
