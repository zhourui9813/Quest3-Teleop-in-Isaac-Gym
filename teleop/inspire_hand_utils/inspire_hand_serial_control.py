import serial
import time

regdict = {
    'ID' : 1000,
    'baudrate' : 1001,
    'clearErr' : 1004,
    'forceClb' : 1009,
    'angleSet' : 1486,
    'forceSet' : 1498,
    'speedSet' : 1522,
    'angleAct' : 1546,
    'forceAct' : 1582,
    'errCode' : 1606,
    'statusCode' : 1612,
    'temp' : 1618,
    'actionSeq' : 2320,
    'actionRun' : 2322
}


def openSerial(port, baudrate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    return ser


def writeRegister(ser, id, add, num, val):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(num + 3) # len
    bytes.append(0x12) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    ser.read_all() # 把返回帧读掉，不处理


def readRegister(ser, id, add, num, mute=False):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(0x04) # len
    bytes.append(0x11) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    bytes.append(num)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    recv = ser.read_all()
    if len(recv) == 0:
        return []
    num = (recv[3] & 0xFF) - 3
    val = []
    for i in range(num):
        val.append(recv[7 + i])
    if not mute:
        print('读到的寄存器值依次为：', end='')
        for i in range(num):
            print(val[i], end=' ')
        print()
    return val


def write6(ser, id, str, val):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet':
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, id, regdict[str], 12, val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')


def read6(ser, id, str):
    if str not in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct', 'errCode', 'statusCode', 'temp']:
        raise ValueError(
            'Function call error, correct way: value of str is \'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'temp\'')

    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'angleAct' or str == 'forceAct':
        val = readRegister(ser, id, regdict[str], 12, True)
        if len(val) < 12:
            # print('No data read')
            raise ValueError('No data read')
        val_act = list()
        for i in range(6):
            val_act.append((val[2*i] & 0xFF) + (val[1 + 2*i] << 8))

    else :
        val_act = readRegister(ser, id, regdict[str], 6, True)
        if len(val_act) < 6:
            # print('No data read')
            raise ValueError('No data read')
    return val_act

if __name__ == '__main__':
    print('打开串口！')
    ser = openSerial('COM3', 115200) # 改成自己的串口号和波特率，波特率默认115200
    print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
    write6(ser, 1, 'speedSet', [100, 100, 100, 100, 100, 100])
    time.sleep(2)
    print('设置灵巧手抓握力度参数！')
    write6(ser, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
    time.sleep(1)
    print('设置灵巧手运动角度参数0，-1为不设置该运动角度！')
    write6(ser, 1, 'angleSet', [0, 0, 0, 0, 400, -1])
    time.sleep(3)
    read6(ser, 1, 'angleAct')
    time.sleep(1)
    print('设置灵巧手运动角度参数1000，-1为不设置该运动角度！')
    write6(ser, 1, 'angleSet', [1000, 1000, 1000, 1000, 400, -1])
    time.sleep(3)
    read6(ser, 1, 'angleAct')
    time.sleep(1)
    read6(ser, 1, 'errCode')
    time.sleep(1)
    print('设置灵巧手动作库序列：8！')
    writeRegister(ser, 1, regdict['actionSeq'], 1, [8])
    time.sleep(1)
    print('运行灵巧手当前序列动作！')
    writeRegister(ser, 1, regdict['actionRun'], 1, [1])
    # writeRegister(ser, 1, regdict['forceClb'], 1, [1])
    # time.sleep(10) # 由于力校准时间较长，请不要漏过这个sleep并尝试重新与手通讯，可能导致插件崩溃
