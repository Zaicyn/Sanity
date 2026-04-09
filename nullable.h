// nullable.h — Analog Nullable Conventions
// ==========================================
// In continuous physics simulation, exact 0.0 only occurs by explicit
// assignment or deliberate cancellation — never by chance. This makes
// 0.0 a natural, collision-free null sentinel for floating-point fields.
//
// Convention:
//   Float fields:   0.0f = uninitialized / inactive / absent
//   Integer indices: -1  = unassigned / no parent
//   Buffer pointers: nullptr = not allocated
//
// These are physically motivated, not arbitrary:
//   - A particle with pump_history = 0.0 hasn't pumped yet
//   - A particle with shell_radius = 0.0 hasn't snapped to a shell
//   - A region with parent_shell = -1 is the bootstrap (no parent)
//   - A buf_* = nullptr means the subsystem isn't allocated
//
// The macros below make intent explicit at call sites without adding
// runtime overhead or struct wrappers to GPU kernel hot paths.

#pragma once

// === Float analog nullables ===
// In continuous simulation, exact 0.0 is maximally information-dense:
// it is simultaneously a validity flag, a physical quantity (zero energy),
// and a state marker (transition from 0 → nonzero is meaningful).
#define IS_NULL_F(val)      ((val) == 0.0f)
#define IS_PRESENT_F(val)   ((val) != 0.0f)

// === Integer sentinel nullables ===
#define INDEX_NULL          (-1)
#define IS_INDEX_NULL(i)    ((i) == INDEX_NULL)
#define IS_INDEX_VALID(i)   ((i) >= 0)

// === Buffer pointer nullables ===
#define IS_BUF_NULL(ptr)    ((ptr) == nullptr)
#define IS_BUF_VALID(ptr)   ((ptr) != nullptr)

// === Topo state nullables ===
// 0x00 = unoccupied (all axes zero), 0xFF = frozen (all axes reserved)
#define TOPO_IS_UNOCCUPIED(state) ((state) == 0x00)
#define TOPO_IS_FROZEN(state)     ((state) == 0xFF)
#define TOPO_IS_ACTIVE(state)     ((state) != 0x00 && (state) != 0xFF)
