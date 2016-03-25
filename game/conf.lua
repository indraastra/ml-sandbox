-- Configuration
function love.conf(t)
    -- The title of the window the game is in (string)
	t.title = "Game"
    -- The LÖVE version this game was made for (string)
	t.version = "0.10.1"
    -- We want our game to be long and thin.
	t.window.width = 400
	t.window.height = 400

	-- For Windows debugging
	t.console = true
end
