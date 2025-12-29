PROJECT_NAME
    ----.vscode
        ----property.json 工程配置文件 用户自定义 (或者存放于工作区的根目录也可)
    ----prj 用于存放工程文件
        ----simulation 用于存放第三方仿真工具运行时的中间文件
            …
        ----factory 用于存放原厂的工程文件 (如：xilinx，efinlix)
            …
    ----user 用于存放用户设计的源文件 用户自定义
        ----ip 用于存放工程ip代码 (厂商工具管理，但由插件搬移至src同级目录)
        ----bd 用于存放工程block design源码 (厂商工具管理，但由插件搬移至src同级目录)
        ----data 主要存放数据文件，以及约束文件
        ----sim 用于存放用户仿真代码
        ----src 用于存放用户的设计源码
        ----sdk 用于存放软件设计，对应xilinx的sdk的设计

# porperty.json 所有属性解说
```json 
{
    // 当前使用的第三方工具链
    "toolChain": "xilinx", 
    "toolVersion" : "2023.2.307",

    // 工程命名 
    // PL : 编程逻辑设计部分即之前的FPGA
    // PS : 处理系统设计部分即之前的SOC
    "prjName": {
        "PL": "template",
        "PS": "template"
    },

    // 自定义工程结构，若无该属性则认为是标准文件结构（详见下文说明）
    "arch" : {
        "structure" : "", // standard | xilinx | custom
        "prjPath" : "",   // 放置工程运行时的中间文件
        "hardware" : {    
            "src"  : "",  // 放置设计源文件，注: src上一级为IP&bd
            "sim"  : "",  // 放置仿真文件，会直接反应在树状结构上
            "data" : ""   // 放置约束、数据文件，约束会自动添加进vivado工程
        },
        "software" : {
            "src"  : "",  // 放置软件设计文件
            "data" : ""   // 放置软件相关数据文件
        }
    },

    // 代码库管理，支持远程和本地两种调用方式（详见下文库管理）
    // 使用UI来进行配置，不建议用户直接更改
    "library" : {
        "state": "", // local | remote
        "hardware" : {
            "common": [], // 插件提供的常见库
            "custom": []  // 用户自己的设计库
        }
    },

    // xilinx的IP仓库，直接添加到vivado的IP repo中
    // 目前支持ADI和ARM提供的IP repo （adi | arm）
    "IP_REPO" : [],

    // 当设计时用到PL+PS即为SOC开发
    // 当其中core不为none的时候即为混合开发
    "soc": {
        "core": "none",
        "bd": "",
        "os": "",
        "app": ""
    },
    
    "device": "none"
}