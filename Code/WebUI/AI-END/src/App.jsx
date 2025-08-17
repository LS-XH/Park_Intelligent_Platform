import { Outlet } from 'react-router-dom';
import styles from './App.module.css';
import LeftArea from './Pages/main/LeftArea';
import { WebSocketProvider } from './Context/wsContext';
import { Buttoncomponent } from './Pages/main/component/begin';
function App() {

  return (
    <WebSocketProvider url="ws://192.168.1.194:5566/ws">
      <div className={styles.container}>
        <div className={styles.leftArea}>
          <LeftArea />
        </div>
        {/* 这里引入图形组的地图 */}
        <div className={`${styles.mapArea} ${styles.newMapStyle}`}>
          <div className={styles.map}>
            <iframe
              src="/Test_Native_WebSocket_(8.14.7.41)/index.html"
              title="Unity WebGL 模型"
              width="100%"
              height="100%"
              style={{
                border: 'none'
              }}
              allowFullScreen
            />
          </div>
          <div className={styles.card}>
            <Buttoncomponent />
          </div>
        </div>
        <div className={styles.outletArea}>
          <Outlet />
        </div>
      </div>
    </WebSocketProvider>
  );
}

export default App;