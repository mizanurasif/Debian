export interface IDeviceInfo {
  name: string;
  platform: string;
  port: string;
  status: 'online' | 'offline' | 'unauthorized' | 'no permissions';
  }

export const getSelectedDevice = async (): Promise<IDeviceInfo> => {
  try {
    const data = await apiGet<
      { status: 'success'; device: IDeviceInfo } | { status: 'error'; message: string }
    >('/api/v1/devices/selectedDevice');

    if (data.status === 'success') {
      return {
        name: data.device.name,
	platform:"common-10.0"
        port: data.device.serial,
        status: data.status,
      } as IDeviceInfo;
    }
    console.log('Error:', data.message);
    return {} as IDeviceInfo;
  } catch {
    return {} as IDeviceInfo;
  }
};

export async function apiGet<T = unknown>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> {
  const ax = await getAxios();
  const res = await ax.get<T>(url, config);
  return res.data;
}


export const getAllDevices = async (): Promise<IDeviceInfo[]> => {
  try {
    const data = await apiGet<
      { status: 'success'; devices: IDeviceInfo[] } | { status: 'error'; message: string }
    >('/api/v1/devices');

    if (data.status === 'success') {
      data.devices = data.devices.map((d) => ({
        ...d,
        capabilityMap: new Map<string, string>(Object.entries(d.capabilityMap!))
      }));
      return data.devices;
    }
    logInfo(`Error:${data.message}`);
    return [];
  } catch {
    return [];
  }
};
