Object = require 'classic/classic'
Timer = require 'hump/timer'

GameCursor = require 'GameCursor'
Ship = require 'Ship'
Trail = require 'Trail'

function love.load(arg)
    -- love.window.setMode(600, 600)

    -- main_canvas = love.graphics.newCanvas(200, 200)
    -- main_canvas:setFilter('nearest', 'nearest')
    love.graphics.setLineStyle('rough')

    timer = Timer()

    game_objects = {}
    cursor = createGameObject('GameCursor', 0, 0)
    ship = createGameObject('Ship', 200, 200)
end

function love.update(dt)
    timer.update(dt)

    for i = #game_objects, 1, -1 do
        local game_object = game_objects[i]
        game_object:update(dt)
        if game_object.dead then table.remove(game_objects, i) end
    end
end

function love.draw(dt)
    -- love.graphics.setCanvas(main_canvas)
    -- love.graphics.clear()

    for _, game_object in ipairs(game_objects) do
        game_object:draw(dt)
    end

    -- love.graphics.setCanvas()
    -- love.graphics.draw(main_canvas, 0, 0, 0, 3, 3)
end

function love.mousepressed(x, y, button)
  if button == 1 then -- 1 = left click
    cursor.dead = true
  end
end

function createGameObject(type, x, y, opts)
    local game_object = _G[type](x, y, opts)
    table.insert(game_objects, game_object)
    return game_object
end

