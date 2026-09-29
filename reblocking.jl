using Statistics, Printf
"""
Reblocking logic for error estimation in DMC
"""
function run_reblocking_loop(data::Vector{Float64}, min_stop_blocks::Int, plateau_tol::Float64=0.01; label::String="energy")
    n_total = length(data)
    i = 1
    block_size = 2^i
    n_blocks = div(n_total, block_size)
    
    # Storage for block sizes and their corresponding errors
    block_sizes = Int[]
    errors = Float64[]
    
    # Perform reblocking for increasing block sizes
    while n_blocks > min_stop_blocks && block_size <= n_total
        error = estimate_error_reblocking(data, block_size)
        push!(block_sizes, block_size)
        push!(errors, error)
        
        # Move to next power of 2
        i += 1
        block_size = 2^i
        n_blocks = div(n_total, block_size)
    end
    
    # Calculate differences between consecutive errors
    diffs = zeros(Float64, length(errors) - 1)
    rel_diffs = zeros(Float64, length(errors) - 1)
    for idx in 1:(length(errors) - 1)
        diffs[idx] = errors[idx + 1] - errors[idx]
        rel_diffs[idx] = abs(diffs[idx]) / errors[idx]
    end
    
    # Write results to file
    open("reblocking_loop_analysis_$label.txt", "w") do f
        @printf(f, "%8s  %12s  %12s  %12s\n", "block_size", "error", "difference", "rel_diff")
        println(f, "-" ^ 55)
        for idx in 1:length(block_sizes)
            if idx < length(block_sizes)
                @printf(f, "%8d  %12.6e  %12.6e  %12.6e\n", 
                    block_sizes[idx], errors[idx], diffs[idx], rel_diffs[idx])
            else
                @printf(f, "%8d  %12.6e  %12s  %12s\n", 
                    block_sizes[idx], errors[idx], "---", "---")
            end
        end
    end
    
    # Find the block size where the error reaches a plateau
    autocorr_block_size, autocorr_error = detect_plateau_max(block_sizes, errors, n_total; tol = plateau_tol, min_blocks = min_stop_blocks)
    return autocorr_block_size, autocorr_error, block_sizes, errors
end


# locate where the plateau is found at
function detect_plateau_max(block_sizes::Vector{Int},
                            errors::Vector{Float64},
                            n_total::Int;
                            tol::Float64 = 0.02,
                            min_blocks::Int = 40)

    imax = argmax(errors)
    emax = errors[imax]

    for i in 1:imax
        block_size = block_sizes[i]
        n_blocks = div(n_total, block_size)

        if n_blocks ≥ min_blocks &&
           errors[i] ≥ (1 - tol) * emax

            return block_size, errors[i]
        end
    end

    # fallback: return maximum itself
    println("Fallback: Returning max error")
    return block_sizes[imax], emax
end

# reblocking for error estimation
function estimate_error_reblocking(data::Vector{Float64}, block_size::Int)
    n_total = length(data)
    n_blocks = div(n_total, block_size)  # integer division 
    
    # Compute block averages
    block_avgs = zeros(n_blocks)
    @inbounds for b in 1:n_blocks
        start_idx = (b - 1) * block_size + 1
        end_idx = b * block_size
        block_avgs[b] = mean(data[start_idx:end_idx])
    end
    
    # Standard error of the block mean
    block_mean = mean(block_avgs)
    block_var = sum((block_avgs .- block_mean).^2) / (n_blocks - 1)  # sample variance with Bessel's correction
    block_err = sqrt(block_var / n_blocks)
    
    return block_err
end