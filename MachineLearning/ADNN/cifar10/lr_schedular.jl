
mutable struct OneCycleSchedular
    # first phase increase to max_lr
    # second phase decrease to ≈ 0
    max_lr # learning rate at pct_start * total_steps
    start_lr
    end_lr
    pct_start # start of second phase
    pct_start_step
    total_steps
    base_momentum # momentum at pct_start * total_steps
    max_momentum # momentum at start and end
    current_step
    function OneCycleSchedular(;max_lr, epochs, batches,
        start_lr=max_lr/25, end_lr=max_lr/10e4, pct_start=0.3,
        base_momentum=0.85, max_momentum=0.95)

        this  = new()

        this.total_steps = epochs * batches
        this.pct_start = pct_start
        this.pct_start_step = this.total_steps * pct_start

        this.start_lr = start_lr
        this.max_lr = max_lr
        this.end_lr = end_lr

        this.base_momentum = base_momentum
        this.max_momentum = max_momentum

        this.current_step = 0

        return this
    end
end

function step!(ocs::OneCycleSchedular)
    ocs.current_step += 1
end

function cos_annealing(f0, f1, pct)
    # pct from 0 to 1
    cos_out = cos(π * pct) + 1 # = 2 if pct = 0; 0 if pct = 1
    return f1 + (f0 - f1) * cos_out / 2
end


function get_lr(ocs::OneCycleSchedular)
    cs = ocs.current_step
    if cs ≤ ocs.pct_start_step
        # first phase: increasing
        pct = cs / ocs.pct_start_step # ∈ [0,1]
        lr0 = ocs.start_lr
        lr1 = ocs.max_lr

        m0 = ocs.base_momentum
        m1 = ocs.max_momentum
    else
        # second phase: decreasing
        pct = (cs - ocs.pct_start_step) / (ocs.total_steps - ocs.pct_start_step) # ∈ [0,1]
        lr0 = ocs.max_lr
        lr1 = ocs.end_lr

        m0 = ocs.max_momentum
        m1 = ocs.base_momentum
    end

    return cos_annealing(lr0, lr1, pct)# , cos_annealing(m0, m1, pct)
end

function plot_schedule(;kw...)
    scheduler = OneCycleSchedular(;kw...)
    lrs = []
    for i in 0:scheduler.total_steps
        push!(lrs, get_lr(scheduler))
        step!(scheduler)
    end
    plot(lrs)
end
