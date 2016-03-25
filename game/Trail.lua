-- in Trail.lua
local Trail = Object:extend()

function Trail:new(x, y, opts)
    self.dead = false
    self.x, self.y = x, y
    self.a = 255
    local opts = opts or {} -- this handles the case where opts is nil
    for k, v in pairs(opts) do self[k] = v end
  
    timer.tween(0.3, self, {r = 0, a = 0}, 'linear',
                function() self.dead = true end)
end

function Trail:update(dt)

end

function Trail:draw()
    love.graphics.setColor({255, 255, 255, self.a})
    love.graphics.circle('fill', self.x, self.y, self.r)
end

return Trail
