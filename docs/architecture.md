# Architecture

## Overview
The package is intended to be structured around four main responsibilities:
- **DAEProblem**: defines the DAE to be solved. The class has two subclasses **FullyimplicitDAEProblem** and **SemiexplicitDAEProblem**.
- **Numerical methods**: implements numerical time-stepping methods to solve DAEs. The initial focus is on SDC and RK methods (but may be expanded some time)
- **TimeStepper**: performs individual time steps using a selected numerical method.
- **Simulation**: advances the solution over a complete time interval and coordinates statistics.  

The overall architecture is currently envisioned as follows:

```text
FullyimplicitDAEProblem                   SemiexplicitDAEProblem
           │                                        │
           └────────────────────┬───────────────────┘
                                │
                                ▼
                            DAEProblem
                                │
                                ▼
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
        SDCMethod                               RKMethod
            │                                        │
            └───────────────────┬────────────────────┘
                           TimeStepper ────── ▶ Logging
                                │
                                ▼
                             TimeStep
                                │
                                ▼
                            Simulation
                                │
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
        DAE Solution                             Statistics
                                                     │
                                                     ▼
                                                  Analysis
```
This architecture is intended to separate the numerical method from the
overall simulation procedure.

## Main components
### DAEProblem
While `DAEProblem` is only a parent class, its child classes `FullyimplicitDAEProblem` and `SemiexplicitDAEProblem` define the mathematical problem to be solved. In the `FullyimplicitDAEProblem` class, the DAE problem is written as an IDE of the form

$$0 = F(t, u(t), u'(t)).$$

Semi-explicit DAEs of the form

$$y'(t) = f(y(t), z(t), t),\qquad \qquad 0 = g(y(t), z(t), t)$$

can be implemented using the `SemiexplicitDAEProblem` class, where the differential and algebraic variables $y,\,z$ and their associated equations are separated. Since semi-explicit DAEs can be written as IDEs, a wrapper function is planned to realize this.

### Numerical methods
Numerical methods implement algorithms for advancing a DAE solution over one time step. The initial focus is on:
- Spectral deferred correction (SDC) methods,
- Runge-Kutta (RK) methods.

The method-specific implementation should remain independent of the overall simulation loop as far as possible.

### TimeStepper
`TimeStepper` is responsible for performing a single time step.
Conceptually:
```text
current state
      │
      ▼
  TimeStepper
      │
      ├── numerical method
      │
      ▼
  new state
```
The `TimeStepper` should therefore provide the interface between the numerical method and the higher-level simulation.

### Simulation
`Simulation` is responsible for advancing the solution over a complete time interval.
Conceptually:
```text
t₀ ──► TimeStepper ──► t₁
t₁ ──► TimeStepper ──► t₂
t₂ ──► TimeStepper ──► t₃
                         ⋮
tₙ₋₁ ─► TimeStepper ──► tₙ
```

### Statistics
During the simulation, statistics DAE-specific and method-specific data shall be logged for later analysis. This includes, for example
- algebraic constraints,
- errors of all variables,
- the solution itself (to check consistency),
- Newton iterations, or general computational work.

The exact design of logging the statistics is still under development.

### Initialization of consistent initial conditions
A future component will compute consistent initial conditions for DAEs for the methods to start.
```text
DAE Problem
     │
     ▼
Consistent Initial Conditions
     │
     ▼
Initial State
     │
     ▼
Simulation
```
This component is not yet implemented.

# Current development status
The current implementation work focus on:
- implementing the first SDC method, namely the SDC-C method from the publicatio that you can find [here](https://link.springer.com/article/10.1007/s10915-026-03385-7).

We strive to keep this document up-to-date.