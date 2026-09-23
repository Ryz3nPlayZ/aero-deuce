# 💫 About Me:
im a student<br>software development

## 🌐 Socials:
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=Instagram&logoColor=white)](https://instagram.com/zemu.liu)

## ⚡ My Fastfetch Prompt — Aero

My daily driver prompt: Ghostty + fish + starship (catppuccin_mocha) + custom Aero-Deuce triangle ASCII for fastfetch, tuned for JetBrainsMono Nerd Font line-spacing.

```text
  *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*=
  :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*+-.
    =%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*=:.         zemuliu@Zemus-MacBook-Air
     :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#*=:.             -------------------------
       =%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+-:.                  OS - macOS Golden Gate 27.0 (26A428) arm64
        :*@@@@@@@@@@@@@@@@@@@@@@@%#+-..                      HOST - MacBook Air (13-inch, M5, 2026)
          =%@@@@@@@@@@@@@@@@@@%+-.                           VER - Darwin 27.0.0
           .*@@@@@@@@@@@@@@@@@+                              UP - 1 day, 9 hours
             -%@@@@@@@@@@@@@@@=                              PKG - 121 (brew), 14 (brew-cask)
              .*@@@@@@@@@@@@@@=                              SH - zsh 5.9
                -%@@@@@@@@@@@@=                              SCR - 3420x2224 @ 2x in 14", 60 Hz
                 .*@@@@@@@@@@@=                              TERM - Ghostty + tmux
                   -%@@@@@@@@@=                              DISK - 330 GiB / 460 GiB (72%) - apfs
                    .*@@@@@@@@=                              IP - 192.168.0.120/24
                      -%@@@@@@=                              BATT - 100% [AC Connected]
                       .*@@@@@=                              PWR - 40W Pwr Adapter 60W Max
                         -%@@@=
                           .#@=
                             -
```

### Setup

**1. Logo — `assets/aero-ascii.txt` (55×19, factor 0.40, sharp `-` tip):**
```text
*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*=
:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*+-.
  =%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*=:.
   :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#*=:.
     =%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+-:.
      :*@@@@@@@@@@@@@@@@@@@@@@@%#+-..
        =%@@@@@@@@@@@@@@@@@@%+-.
         .*@@@@@@@@@@@@@@@@@+
           -%@@@@@@@@@@@@@@@=
            .*@@@@@@@@@@@@@@=
              -%@@@@@@@@@@@@=
               .*@@@@@@@@@@@=
                 -%@@@@@@@@@=
                  .*@@@@@@@@=
                    -%@@@@@@=
                     .*@@@@@=
                       -%@@@=
                         .#@=
                           -
```

**2. Fastfetch — `~/.config/fastfetch/aero.jsonc`:**
```jsonc
{
  "logo": {
    "type": "file-raw",
    "source": "/Users/zemuliu/Programming/aero-deuce/assets/aero-ascii.txt",
    "padding": { "top": 1, "left": 2, "right": 3 }
  },
  "display": { "separator": "- " },
  "modules": [
    "break", "break", "break",
    { "type": "title" },
    { "type": "separator" },
    { "type": "os", "key": " OS ", "keyColor": "red" },
    { "type": "host", "key": " HOST ", "keyColor": "green" },
    { "type": "kernel", "key": " VER ", "keyColor": "yellow" },
    { "type": "uptime", "key": " UP ", "keyColor": "blue" },
    { "type": "packages", "key": " PKG ", "keyColor": "magenta" },
    { "type": "shell", "key": " SH ", "keyColor": "cyan" },
    { "type": "display", "key": " SCR ", "keyColor": "white" },
    { "type": "terminal", "key": " TERM ", "keyColor": "yellow" },
    { "type": "disk", "key": " DISK ", "keyColor": "blue" },
    { "type": "localip", "key": " IP ", "keyColor": "magenta" },
    { "type": "battery", "key": " BATT ", "keyColor": "cyan" },
    { "type": "poweradapter", "key": " PWR ", "keyColor": "white" },
    "break",
    { "type": "colors", "symbol": "circle" }
  ]
}
```

**3. Run it:**
```bash
fastfetch -c ~/.config/fastfetch/aero.jsonc
# make default:
cp ~/.config/fastfetch/aero.jsonc ~/.config/fastfetch/config.jsonc
```

Fish auto-runs it (`config.fish`):
```fish
if status is-interactive
    if command -v fastfetch &>/dev/null
        fastfetch
    end
end
```

Stack: Ghostty (`JetBrainsMono Nerd Font` 14,ekaneskode) + fish + starship `catppuccin_mocha` + zoxide + eza.

Why this ratio? Terminal cells aren't square + line-spacing stretches height. `0.40` factor with hand-sharpened `-` tip keeps both legs sharp. Half-blocks still show line gaps on this setup, so classic `@%#*+=-:. ` wins.

---
# 💻 Tech Stack:
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white) ![Swift](https://img.shields.io/badge/swift-F54A2A?style=for-the-badge&logo=swift&logoColor=white) ![Bash Script](https://img.shields.io/badge/bash_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB) ![Next JS](https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# 📊 GitHub Stats:
![](https://github-readme-stats.shion.dev/api?username=Ryz3nPlayZ&theme=dark&hide_border=false&include_all_commits=true&count_private=true)<br/>
![](https://streak-stats.demolab.com/?user=Ryz3nPlayZ&theme=dark&hide_border=false)<br/>
![](https://github-readme-stats.shion.dev/api/top-langs/?username=Ryz3nPlayZ&theme=dark&hide_border=false&include_all_commits=true&count_private=true&layout=compact)

### ✍️ Random Dev Quote
![](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=radical)

---
[![](https://komarev.com/ghpvc/?username=Ryz3nPlayZ&icon=0&color=0)](https://visitcount.itsvg.in)
