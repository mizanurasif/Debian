pub enum DeviceStatus {
    Online,
    Offline,
    Unauthorized,
    #[serde(rename = "no permissions")]
    NoPermissions,
}
pub struct DeviceInfo {
    pub serial: String,
    pub name: String,
    pub status: DeviceStatus,
    pub platform: String,
    // Map<string, string>
    pub capability_map: Option<HashMap<String, String>>,
}
async fn fetch_selected_device() -> Result<DeviceInfo, String> {
    let client = Client::new();

    let res = client
        .get("http://127.0.0.1:58793/api/v1/devices/selectedDevice")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let json: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;

    if json["status"] == "success" {
        let device = &json["device"];

        Ok(DeviceInfo {
            name: device["name"].as_str().unwrap_or("").to_string(),
            platform: "common-10.0".to_string(),
            serial: device["serial"].as_str().unwrap_or("").to_string(),
            status: DeviceStatus::Online,
            capability_map: None,
        })
    } else {
        Err(json["message"].as_str().unwrap_or("Unknown error").to_string())
    }

{
  "status": "success",
  "devices": [
    {
      "serial": "emulator-26101",
      "name": "T-10.0-x86_64",
      "status": "device",
      "capabilityMap": {
        "secure_protocol": "disabled",
        "intershell_support": "enabled",
        "filesync_support": "pushpull",
        "usbproto_support": "enabled",
        "sockproto_support": "enabled",
        "syncwinsz_support": "enabled",
        "sdbd_rootperm": "disabled",
        "rootonoff_support": "enabled",
        "encryption_support": "disabled",
        "zone_support": "disabled",
        "multiuser_support": "enabled",
        "cpu_arch": "x86_64",
        "core_abi": "x86_64",
        "sdk_toolpath": "/home/owner/share/tmp/sdk_tools",
        "profile_name": "common",
        "vendor_name": "Tizen",
        "can_launch": "unknown",
        "device_name": "Tizen",
        "platform_version": "10.0",
        "product_version": "unknown",
        "sdbd_version": "2.2.31",
        "sdbd_plugin_version": "unknown",
        "sdbd_cap_version": "1.0",
        "log_enable": "disabled",
        "log_path": "/home/owner/share/sdbdlog",
        "appcmd_support": "enabled",
        "appid2pid_support": "enabled",
        "pkgcmd_debugmode": "enabled",
        "netcoredbg_support": "enabled",
        "architecture": "64"
      }
    },
    {
      "serial": "emulator-26111",
      "name": "Tizen_Emulator",
      "status": "device",
      "capabilityMap": {
        "secure_protocol": "disabled",
        "intershell_support": "enabled",
        "filesync_support": "pushpull",
        "usbproto_support": "enabled",
        "sockproto_support": "enabled",
        "syncwinsz_support": "enabled",
        "sdbd_rootperm": "disabled",
        "rootonoff_support": "enabled",
        "encryption_support": "disabled",
        "zone_support": "disabled",
        "multiuser_support": "enabled",
        "cpu_arch": "x86_64",
        "core_abi": "x86_64",
        "sdk_toolpath": "/home/owner/share/tmp/sdk_tools",
        "profile_name": "common",
        "vendor_name": "Tizen",
        "can_launch": "unknown",
        "device_name": "Tizen",
        "platform_version": "10.0",
        "product_version": "unknown",
        "sdbd_version": "2.2.31",
        "sdbd_plugin_version": "unknown",
        "sdbd_cap_version": "1.0",
        "log_enable": "disabled",
        "log_path": "/home/owner/share/sdbdlog",
        "appcmd_support": "enabled",
        "appid2pid_support": "enabled",
        "pkgcmd_debugmode": "enabled",
        "netcoredbg_support": "enabled",
        "architecture": "64"
      }
    }
  ]
}
