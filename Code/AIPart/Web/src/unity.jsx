const unity = () => {
    return (
        <div style={{
            width: '100vw',
            height: '100vh',
            position: 'fixed',
            top: 0,
            left: 0,
            zIndex: 1000
        }}>
            <iframe
                src="/TestProgram/index.html"
                title="Unity WebGL 模型"
                width="100%"
                height="100%"
                style={{ border: 'none' }}
                allowFullScreen
            />
        </div>
    );
};

export default unity;