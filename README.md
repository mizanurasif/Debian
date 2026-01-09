i want my log in this format
function formatLog(logString: string): Log {
    let logArray = logString.split(" ");
    if (logArray.length < 3) {
        return {} as Log;
    }
    let time = logArray[1].slice(0, -5);
    let level = "";
    switch (logArray[2][0]) {
        case 'I':
            level = "Info"
            break;
        case 'D':
            level = "Debug"
            break;
        case 'W':
            level = "Warning"
            break;
        case 'E':
            level = "Error"
            break;
        default:
            level = "Unknown"
            break;
    }
    let tag = logArray[2].slice(2);
    let pid_start = logString.indexOf("(P")
    let pid = logString.slice(pid_start + 2, pid_start + 7);
    let tid_start = logString.indexOf(", T");
    let tid = logString.slice(tid_start + 3, tid_start + 8);

    let f = logString.indexOf("):", tid_start)
    let message = logString.slice(f + 2);
    return {
        time: time,
        level: level,
        pid: pid,
        tid: tid,
        tag: tag,
        message: message
    }
}
in my js code its parse one line form log a push it to frontend. is there better way. this this
01-09 14:12:39.848+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.848963][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:39.849+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.849015][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:39.849+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.849075][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:39.849+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.849178][2445][drm_hwc_commit 702] ==============COMMIT=================================
01-09 14:12:39.974+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.974930][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:39.975+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.975090][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:39.975+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.975159][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:39.975+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.975209][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:39.975+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.975274][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:39.975+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.975340][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976199][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976250][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976295][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976339][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976388][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976447][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:39.976+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3704.976546][2445][drm_hwc_commit 702] ==============COMMIT=================================
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098219][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098312][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098360][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098405][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098458][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098518][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:40.098+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.098618][2445][drm_hwc_commit 702] ==============COMMIT=================================
01-09 14:12:40.224+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.224620][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:40.224+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.224695][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:40.224+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.224740][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:40.225+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.225055][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:40.225+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.225114][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:40.225+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.225220][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339081][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339154][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339206][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339257][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339317][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339462][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:40.339+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.339590][2445][drm_hwc_commit 702] ==============COMMIT=================================
01-09 14:12:40.370+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.370904][2445][drm_hwc_validate 621] ==============Validate=================================
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.370976][2445][_print_validate_result 183] window(0x5614341d6500) CLIENT -> CLIENT : lzpos(0) -- {Cursor} on TARGET WINDOW
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.371015][2445][_print_validate_result 183] window(0x5614341848b0) DEVICE -> CLIENT : lzpos(0) -- {/usr/apps/org.tizen.taskbar/bin/TaskBar.dll} on TARGET WINDOW
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.371053][2445][_print_validate_result 183] window(0x5614341ff700) DEVICE -> CLIENT : lzpos(0) -- {UNKNOWN} on TARGET WINDOW
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.371097][2445][drm_hwc_accept_validation 677] ==============Accept Changes Done=================================
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.371203][2445][_drm_hwc_prepare_commit 338] lzpos(0) : SET
01-09 14:12:40.371+0900 I/TDM_BACKEND (P 2445, T 2445): [ 3705.371294][2445][drm_hwc_commit 702] ==============COMMIT=================================
01-09 14:12:41.385+0900 I/E20_MONITOR (P20467, T20467): ## Enlightenment (PID=2445, STAT=S, WCHAN=poll_schedule_timeout) is running ! ##
01-09 14:12:42.004+0900 D/CONNMAN  (P 2473, T 2473): src/ntp.c:send_timeout() send timeout 4 (retries 1)
01-09 14:12:42.010+0900 D/WIFI_TETHERING_MANAGER (P 2415, T 2415): [__initialize_monitor:49] Initializing WiFi tethering manager.
01-09 14:12:42.010+0900 D/WIFI_TETHERING_MANAGER (P 2415, T 2415): [__initialize_monitor:51] There is necessary to initialize Wi-Fi handle.
01-09 14:12:42.010+0900 I/WIFI_MANAGER (P 2415, T 2415): wifi_manager.c: wifi_manager_initialize(99) > Enter
01-09 14:12:42.010+0900 E/WIFI_MANAGER (P 2415, T 2415): wifi_internal.c: _wifi_check_feature_supported(3552) > http://tizen.org/feature/network.wifi Feature is not supported
01-09 14:12:42.010+0900 E/WIFI_TETHERING_MANAGER (P 2415, T 2415): wifi-tethering-wifi.c: __initialize_wifi_handle(256) > [__initialize_wifi_handle:256] wifi_manager_initialize failed. [WIFI_MANAGER_ERROR_NOT_SUPPORTED]
01-09 14:12:42.011+0900 E/WIFI_DIRECT (P 2415, T 2415): wifi-direct-client-proxy.c: wifi_direct_initialize(1017) > http://tizen.org/feature/network.wifi.direct feature is disabled
01-09 14:12:42.011+0900 E/WIFI_TETHERING_MANAGER (P 2415, T 2415): wifi-tethering-p2p.c: __initialize_p2p_handle(259) > [__initialize_p2p_handle:259] wifi_direct_initialize failed. [UNKNOWN]
01-09 14:12:42.024+0900 I/CKM      (P 2211, T 2340): [glib-logic.cpp:133] watchdogMsgSender(): aw_notify success!
01-09 14:12:44.394+0900 I/E20_MONITOR (P20481, T20481): ## Enlightenment (PID=2445, STAT=S, WCHAN=poll_schedule_timeout) is running ! ##
01-09 14:12:44.410+0900 I/PULSEAUDIO (P19108, T19103): stream-manager.c: monitoring_cb(3489) > monitoring callback invoked, event(0x55da53c642f0)
01-09 14:12:44.410+0900 W/PULSEAUDIO (P19108, T19103): core.c: pa_core_dump_sink_inputs(695) > No sink-input to dump
01-09 14:12:44.410+0900 W/PULSEAUDIO (P19108, T19103): core.c: pa_core_dump_source_outputs(709) > No source-output to dump
01-09 14:12:44.410+0900 W/PULSEAUDIO (P19108, T19103): stream-manager.c: dump_ducking(3466) > No ducking to dump
01-09 14:12:47.406+0900 I/E20_MONITOR (P20495, T20495): ## Enlightenment (PID=2445, STAT=S, WCHAN=poll_schedule_timeout) is running ! #
