#if (VER < EncodeVer(6,7,0)) || (VER >= EncodeVer(6,7,1))
  #error Build requires the verified Inno Setup 6.7.0 compiler
#endif

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#ifndef BundleDir
  #error BundleDir must identify the verified onedir payload
#endif
#ifndef OutputDir
  #error OutputDir must identify the owned artifact directory
#endif
#ifndef HereAppId
  #define HereAppId "{{94BE3378-6B95-48A1-BD13-C534746C7319}"
#endif
#ifndef HereGroup
  #define HereGroup "here"
#endif

[Setup]
AppId={#HereAppId}
AppName=here
AppVersion={#AppVersion}
AppPublisher=here contributors
DefaultDirName={localappdata}\Programs\here
DefaultGroupName={#HereGroup}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
OutputDir={#OutputDir}
OutputBaseFilename=here-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
AppMutex=here.application.94BE3378-6B95-48A1-BD13-C534746C7319
CloseApplications=no
RestartApplications=no
UninstallDisplayIcon={app}\here.exe
SetupLogging=yes
DisableReadyPage=no

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\here"; Filename: "{app}\here.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall here"; Filename: "{uninstallexe}"

; No Run/startup tasks, services, capture, or user-data uninstall-delete entries.
