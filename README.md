# DeLi
Robot Programming Team4  
김지성 김민서 임성우 지선우

## Environment
- **Robot**: TurtleBot3 Burger
- **SBC**: Raspberry Pi 4
- **OS**: Ubuntu 24.04
- **Middleware**: ROS2 Jazzy
- **System**: Automated delivery system (receipt recognition → navigation to destination → bell pressing → return home)

## 📁 Project Structure

```
src/
├── delivery_interfaces/          # ROS 2 interface definitions
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── action/
│   │   └── ApproachBell.action   # Doorbell approach action
│   └── srv/
│       ├── AnalyzeReceipt.srv    # Receipt analysis service
│       └── VerifyBell.srv        # Doorbell verification service
│
├── delivery_nav/                 # Autonomous navigation package
│   ├── package.xml
│   ├── setup.cfg
│   ├── setup.py
│   ├── delivery_nav/
│   │   ├── __init__.py
│   │   ├── nav_node.py           # Navigation node
│   │   ├── red_detector.py       # Red object detection
│   │   └── config.py             # Configuration management
│   └── resource/
│
├── delivery_sm/                  # State machine package
│   ├── package.xml
│   ├── setup.cfg
│   ├── setup.py
│   ├── delivery_sm/
│   │   ├── __init__.py
│   │   ├── state_machine_node.py # State machine implementation
│   │   └── receipt_ui.py         # Receipt UI
│   ├── launch/
│   │   └── delivery.launch.py    # Launch file
│   └── resource/
│
├── delivery_vlm/                 # Vision Language Model package
│   ├── package.xml
│   ├── setup.cfg
│   ├── setup.py
│   ├── delivery_vlm/
│   │   ├── __init__.py
│   │   ├── vlm_node.py           # VLM main node
│   │   ├── bell_vlm.py           # Doorbell detection VLM
│   │   └── config.py             # Model configuration
│   └── resource/
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Package Roles

| Package | Type | Role |
|---------|------|------|
| `delivery_interfaces` | ament_cmake | Service/Action interface definitions |
| `delivery_vlm` | ament_python | Receipt/Doorbell recognition via VLM |
| `delivery_nav` | ament_python | Autonomous navigation and path planning |
| `delivery_sm` | ament_python | State machine and system coordination |

## Node Communication Flow

```
[state_machine_node]
   │  (Receipt capture GUI)
   ├── srv  /analyze_receipt ─────▶ [vlm_node]
   ├── action /approach_bell ─────▶ [nav_node]
   │                                    │
   │                                    ├── srv /verify_bell ──▶ [vlm_node]
   │                                    └── action navigate_to_pose ──▶ [Nav2]
   └── topic /delivery_status ◀──────── [nav_node]
```

## 🚀 How to Run

### 1. Prerequisites

```bash
sudo apt update
sudo apt install ros-jazzy-cv-bridge ros-jazzy-nav2-msgs
pip install -r requirements.txt
```

### 2. Build

```bash
cd ~/turtlebot3_ws
# Build delivery_interfaces first
colcon build --packages-select delivery_interfaces
source install/setup.bash
# Build remaining packages
colcon build --symlink-install
source install/setup.bash
```

### 3. Execution

```bash
cd ~/turtlebot3_ws
source install/setup.bash
export ANTHROPIC_API_KEY=sk-ant-...

# Start the system
ros2 launch delivery_sm delivery.launch.py
```

## 💻 Usage Flow

1. **Capture Receipt**: Point receipt at Receipt window and press **SPACE** → VLM analysis
2. **Confirm Result**: **G**(proceed) / **R**(recapture) / **Q·ESC**(exit)
3. **Automatic Delivery**: Robot navigates to destination and presses doorbell
4. **Return Home**: Press **F** to automatically return to home position

## ⚙️ Configuration

Modify in `delivery_nav/delivery_nav/config.py`:
- `HOME_POSITION`: Home location coordinates
- `GUARD_ROOM_GOAL`: Guard room coordinates
- `ROOM_GOALS`: Coordinates and orientations for each room number

## 🔍 Debugging

```bash
# Check interfaces
ros2 interface show delivery_interfaces/srv/AnalyzeReceipt
ros2 interface show delivery_interfaces/action/ApproachBell

# Check nodes and communication
ros2 node list
ros2 service list | grep -E 'analyze|verify'
ros2 action list | grep approach
ros2 topic echo /delivery_status
```
