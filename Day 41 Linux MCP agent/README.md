# 🐧 Linux AI System Administrator — MCP Multi-Agent System

> Ask "Why is my server slow?" instead of chaining ten different Linux commands.

An AI assistant that manages one or more Linux machines **from Windows**, using the **Model Context Protocol (MCP)** to expose system administration capabilities as tools that a LangGraph-powered agent can call on demand.

Instead of manually running:

```bash
df -h
top
docker ps
journalctl -u nginx
systemctl status postgresql
```

...you just ask:

```
"Why is my server slow?"
```

The agent figures out which MCP servers to call, gathers the evidence, and explains the root cause in plain English.

---

## 💡 Why This Project

This project isn't just "an AI that runs shell commands" — it's a demonstration of the real value of MCP: **one AI agent coordinating many specialized, independently-testable tools**, instead of hardcoding every capability into a single monolithic application.

Each Linux subsystem (Docker, systemd, logs, networking, packages, GPU) gets its own focused MCP server. The LangGraph supervisor decides, per query, which servers to call and in what order — then synthesizes the results into a human answer.

---

## 🏗️ Architecture

```
                    Windows PC
              (LangGraph Supervisor Agent)

                       │
                MCP Client Layer
                       │
      ┌────────┬────────┬────────┬────────┐
      │        │        │        │
   SSH MCP  Docker   System    Logs
    Server    MCP      MCP      MCP
      │        │        │        │
      └────────┴────────┴────────┘
                 Ubuntu Server
```

Only the agent runs on Windows. The Linux machine simply **exposes** the functionality the agent needs — each capability lives behind its own MCP server.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **AI / Orchestration** | LangGraph, LangChain, MCP Python SDK, Ollama / OpenAI-compatible LLM, Redis *(optional)*, PostgreSQL *(optional)* |
| **Client (Windows)** | Python, VS Code, Git |
| **Target (Linux)** | Ubuntu Server, Docker, SSH, systemd |

**Getting a Linux target:**
- **Recommended:** A VM (VMware Workstation Player, VirtualBox, or Hyper-V) — safe to break, snapshot-friendly.
- An old laptop running Ubuntu Server.
- A cheap VPS (DigitalOcean, Hetzner, Oracle Cloud Free Tier).

---

## 🧩 MCP Servers & Tools

| Phase | Server | Tools |
|---|---|---|
| 1 | **SSH** | `run_command`, `list_directory`, `read_file`, `write_file`, `download_file`, `upload_file` |
| 2 | **System** | `get_cpu_usage`, `get_memory_usage`, `get_disk_usage`, `get_network_usage`, `get_logged_users`, `get_uptime` |
| 3 | **Docker** | `list_containers`, `start_container`, `stop_container`, `restart_container`, `container_logs`, `container_stats` |
| 4 | **Systemd** | `list_services`, `restart_service`, `stop_service`, `start_service`, `service_status` |
| 5 | **Logs** | `read_journal`, `search_logs`, `tail_logs` |
| 6 | **Filesystem** | `find_file`, `delete_file`, `copy_file`, `move_file`, `search_text` |
| 7 | **Package Manager** | `install_package`, `remove_package`, `update_packages`, `list_updates` |
| 8 | **Network** | `ping_host`, `check_port`, `show_routes`, `dns_lookup`, `network_interfaces` |
| 9 | **GPU** *(optional, NVIDIA only)* | `gpu_status`, `gpu_temperature`, `gpu_processes`, `gpu_memory` |
| 10 | **Supervisor (LangGraph)** | Routes the query to the right combination of servers above and synthesizes a final answer |

The Supervisor never executes commands itself — it **selects and delegates** to the appropriate MCP server(s), then combines the results.

---

## 🔍 Example Workflows

**"Why is my server slow?"**
1. `get_cpu_usage()` → `get_memory_usage()` → `get_disk_usage()` → `container_stats()`
2. Summarizes findings into a single explanation.

**"Why isn't my API working?"**
1. Check service status → read recent logs → check if port 8080 is listening → verify disk space.
2. Explain the likely cause.

**"Restart all unhealthy Docker containers."**
1. List containers → filter unhealthy ones → restart them → confirm the outcome.

---

## 📁 Project Structure

```
linux-admin-agent/
│
├── agent/
│   ├── supervisor.py
│   ├── planner.py
│   ├── state.py
│   └── prompts.py
│
├── mcp_servers/
│   ├── ssh_server/
│   ├── system_server/
│   ├── docker_server/
│   ├── logs_server/
│   ├── systemd_server/
│   ├── network_server/
│   ├── filesystem_server/
│   └── package_server/
│
├── shared/
│   ├── ssh_client.py
│   ├── config.py
│   └── utils.py
│
├── tests/
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 🗺️ Development Roadmap

- [ ] Set up an Ubuntu VM (VirtualBox / VMware) on the Windows host
- [ ] Install & configure `openssh-server` on the VM; verify connection from Windows
- [ ] Build the SSH MCP server (`run_command`) and validate with MCP Inspector
- [ ] Refactor shared SSH connection logic into a common module
- [ ] Build the remaining focused servers: System, Docker, Systemd, Logs, Filesystem, Network
- [ ] Build the LangGraph supervisor that routes to the right MCP tools
- [ ] Add memory (Redis / PostgreSQL) for follow-up questions like *"What about yesterday?"*
- [ ] Package everything with Docker Compose + write setup docs

See `roadmap.svg` for a visual version of this plan.

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/linux-admin-agent.git
cd linux-admin-agent
# set up each MCP server per its own README under mcp_servers/
```

*(Detailed per-server setup instructions to be added as each phase is built.)*

---

