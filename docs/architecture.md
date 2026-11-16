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