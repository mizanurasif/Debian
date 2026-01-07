export interface IDeviceInfo {
  serial: string;
  name: string;
  status: 'online' | 'offline' | 'unauthorized' | 'no permissions';
  platform: string;
  model?: string;
  dpi?: string;
  resolution?: string;
  capabilityMap?: Map<string, string>;
}

export interface IDeviceLogConfig {
  serial: string;
  format: 'threadtime' | 'long' | 'time' | 'raw';
  filter: string;
  isLogging: boolean;
}

export interface IDeviceData {
  devices: IDeviceInfo[];
  selectedDevice?: string;
  logConfigs: IDeviceLogConfig[];
  tizenStudioPath?: string;
  sdbPath?: string;
  isMonitoring: boolean;
}

export interface IDeviceActionData {
  action: 'select' | 'start-log' | 'stop-log' | 'refresh' | 'start-remote-log' | 'update-log-format' | 'update-log-filter';
  deviceSerial?: string;
  logFormat?: 'threadtime' | 'long' | 'time' | 'raw';
  logFilter?: string;
  data?: Record<string, unknown>;
}
