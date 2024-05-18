package ort

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

type GPU_Stats struct {
	DeviceID int    // GPU編號
	Total    int    // MiB
	Used     int    // MiB
	Free     int    // MiB
	Name     string // GPU名稱
}

func GPU_stats() ([]GPU_Stats, error) {
	bin, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, fmt.Errorf("lookup nvidia-smi error: %s:", err)
	}
	// 執行 nvidia-smi 命令
	cmd := exec.Command(bin, "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits")
	out := bytes.NewBufferString("")
	cmd.Stdout = out
	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("executing nvidia-smi error: %s:", err)
	}

	// 解析輸出
	r := csv.NewReader(out)
	gpus, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse csv of nvidia-smi error: %s:", err)
	}
	for i, j := 0, len(gpus)-1; i < j; i, j = i+1, j-1 {
		gpus[i], gpus[j] = gpus[j], gpus[i]
	}

	stats := []GPU_Stats{}
	id := 0
	for _, gpu := range gpus {
		if len(gpu) < 4 {
			continue
		}
		name := strings.TrimSpace(gpu[0])
		total, err := strconv.Atoi(strings.TrimSpace(gpu[1]))
		if err != nil {
			return nil, fmt.Errorf("parse total memory error: %s:", err)
		}
		used, err := strconv.Atoi(strings.TrimSpace(gpu[2]))
		if err != nil {
			return nil, fmt.Errorf("parse used memory error: %s:", err)
		}
		free, err := strconv.Atoi(strings.TrimSpace(gpu[3]))
		if err != nil {
			return nil, fmt.Errorf("parse free memory error: %s:", err)
		}
		stats = append(stats, GPU_Stats{
			DeviceID: id,
			Name:     name,
			Total:    total,
			Used:     used,
			Free:     free,
		})
		id++
	}
	return stats, nil
}
