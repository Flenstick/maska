; Inno Setup: установщик «Маски». Версию передаёт CI:  iscc /DAppVersion=1.0.0 packaging\windows\maska.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F3C8E2A-4B1D-4C7E-9A55-2D8F0B7E91C4}
AppName=Маска
AppVersion={#AppVersion}
AppVerName=Маска {#AppVersion}
AppPublisher=MrFlen
AppPublisherURL=https://github.com/Flenstick/maska
AppSupportURL=https://github.com/Flenstick/maska/issues
DefaultDirName={localappdata}\Programs\Maska
DefaultGroupName=Маска
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\dist
OutputBaseFilename=Maska-Setup-{#AppVersion}
SetupIconFile=maska.ico
UninstallDisplayIcon={app}\Maska.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\Maska\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Маска"; Filename: "{app}\Maska.exe"
Name: "{autodesktop}\Маска"; Filename: "{app}\Maska.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Maska.exe"; Description: "{cm:LaunchProgram,Маска}"; Flags: nowait postinstall skipifsilent
