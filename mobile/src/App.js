import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Image,
  Modal,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { formatDateTime } from "./utils/format";

const DEFAULT_API_BASE =
  process.env.EXPO_PUBLIC_API_BASE_URL || "http://localhost:8000";

const fetchJson = async (url, options = {}) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "リクエストに失敗しました");
  }
  return data;
};

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [history, setHistory] = useState([]);
  const [results, setResults] = useState([]);
  const [activeEntry, setActiveEntry] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [sendingChat, setSendingChat] = useState(false);

  const apiBase = DEFAULT_API_BASE.replace(/\/$/, "");

  const latestResults = useMemo(() => results.slice(0, 6), [results]);

  const loadModels = useCallback(async () => {
    try {
      const data = await fetchJson(`${apiBase}/models`);
      setModels(data.models || []);
      setSelectedModel(data.default || "");
    } catch (err) {
      Alert.alert("モデル取得エラー", err.message);
    }
  }, [apiBase]);

  const loadHistory = useCallback(async () => {
    try {
      const data = await fetchJson(`${apiBase}/history`);
      if (data.success) {
        setHistory(data.history || []);
      }
    } catch (err) {
      Alert.alert("履歴取得エラー", err.message);
    }
  }, [apiBase]);

  useEffect(() => {
    loadModels();
    loadHistory();
  }, [loadModels, loadHistory]);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("権限が必要です", "写真へのアクセスを許可してください。");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9
    });

    if (!result.canceled && result.assets?.length) {
      uploadImage(result.assets[0]);
    }
  };

  const uploadImage = async (asset) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", {
        uri: asset.uri,
        name: asset.fileName || `photo-${Date.now()}.jpg`,
        type: asset.mimeType || "image/jpeg"
      });

      const query = selectedModel
        ? `?model_name=${encodeURIComponent(selectedModel)}`
        : "";
      const response = await fetch(`${apiBase}/diagnose${query}`, {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json"
        }
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.detail || "診断に失敗しました");
      }

      setResults((prev) => [data, ...prev]);
      await loadHistory();
    } catch (err) {
      Alert.alert("診断エラー", err.message);
    } finally {
      setUploading(false);
    }
  };

  const openEntry = async (entryId) => {
    try {
      const data = await fetchJson(`${apiBase}/history/${entryId}`);
      if (data.success) {
        setActiveEntry(data.entry);
      }
    } catch (err) {
      Alert.alert("詳細取得エラー", err.message);
    }
  };

  const sendChat = async () => {
    if (!activeEntry || !chatInput.trim()) return;
    setSendingChat(true);
    try {
      const data = await fetchJson(`${apiBase}/chat/${activeEntry.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: chatInput.trim() })
      });

      if (data.success) {
        const now = new Date().toISOString();
        setActiveEntry((prev) => ({
          ...prev,
          chat: [
            ...(prev.chat || []),
            { role: "user", message: chatInput.trim(), timestamp: now },
            { role: "assistant", message: data.reply, timestamp: now }
          ]
        }));
        setChatInput("");
      }
    } catch (err) {
      Alert.alert("チャット送信エラー", err.message);
    } finally {
      setSendingChat(false);
    }
  };

  const clearHistory = async () => {
    Alert.alert("履歴削除", "全履歴を削除します。", [
      { text: "キャンセル", style: "cancel" },
      {
        text: "削除",
        style: "destructive",
        onPress: async () => {
          try {
            const data = await fetchJson(`${apiBase}/clear-history`, {
              method: "POST"
            });
            if (data.success) {
              setHistory([]);
              setActiveEntry(null);
              setResults([]);
            }
          } catch (err) {
            Alert.alert("削除エラー", err.message);
          }
        }
      }
    ]);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.hero}>
          <Text style={styles.kicker}>PlantLLM Mobile</Text>
          <Text style={styles.title}>植物診断をポケットに。</Text>
          <Text style={styles.subtitle}>
            画像を選ぶだけで植物の種類と健康状態を診断します。
          </Text>
          <View style={styles.heroMeta}>
            <View style={styles.badge}>
              <Text style={styles.badgeTitle}>モデル</Text>
              <Text style={styles.badgeText}>
                {models.find((m) => m.name === selectedModel)?.description ||
                  "読み込み中"}
              </Text>
            </View>
            <View style={styles.badge}>
              <Text style={styles.badgeTitle}>履歴</Text>
              <Text style={styles.badgeText}>{history.length}件</Text>
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>画像を診断する</Text>
          <Text style={styles.cardSub}>1枚ずつ素早く診断します。</Text>
          <Pressable
            style={[styles.primaryButton, uploading && styles.disabledButton]}
            onPress={pickImage}
            disabled={uploading}
          >
            <Text style={styles.primaryButtonText}>
              {uploading ? "診断中..." : "画像を選択"}
            </Text>
          </Pressable>
          <Text style={styles.helperText}>
            ※ 実機で利用する場合は API のベースURLを設定してください。
          </Text>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>最新の診断</Text>
            <Text style={styles.sectionMeta}>{results.length}件</Text>
          </View>
          {latestResults.length === 0 ? (
            <Text style={styles.muted}>まだ診断結果がありません。</Text>
          ) : (
            latestResults.map((result) => (
              <Pressable
                key={result.entry_id}
                style={styles.resultCard}
                onPress={() => openEntry(result.entry_id)}
              >
                <Image
                  source={{ uri: `${apiBase}${result.image?.url}` }}
                  style={styles.resultImage}
                />
                <View style={styles.resultBody}>
                  <Text style={styles.resultMeta}>
                    {formatDateTime(result.timestamp)}
                  </Text>
                  <Text style={styles.resultText} numberOfLines={3}>
                    {result.diagnosis}
                  </Text>
                </View>
              </Pressable>
            ))
          )}
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>診断履歴</Text>
            <TouchableOpacity onPress={clearHistory}>
              <Text style={styles.linkText}>削除</Text>
            </TouchableOpacity>
          </View>
          {history.length === 0 ? (
            <Text style={styles.muted}>履歴がありません。</Text>
          ) : (
            history.slice(0, 6).map((entry) => (
              <Pressable
                key={entry.id}
                style={styles.historyItem}
                onPress={() => openEntry(entry.id)}
              >
                <Image
                  source={{ uri: `${apiBase}${entry.image?.url}` }}
                  style={styles.historyThumb}
                />
                <View style={styles.historyBody}>
                  <Text style={styles.resultMeta}>
                    {formatDateTime(entry.timestamp)}
                  </Text>
                  <Text style={styles.resultText} numberOfLines={2}>
                    {entry.diagnosis}
                  </Text>
                </View>
              </Pressable>
            ))
          )}
        </View>
      </ScrollView>

      <Modal visible={!!activeEntry} animationType="slide" onRequestClose={() => setActiveEntry(null)}>
        <SafeAreaView style={styles.modalContainer}>
          <ScrollView contentContainerStyle={styles.modalContent}>
            <TouchableOpacity onPress={() => setActiveEntry(null)}>
              <Text style={styles.linkText}>閉じる</Text>
            </TouchableOpacity>
            {activeEntry ? (
              <>
                <Image
                  source={{ uri: `${apiBase}${activeEntry.image?.url}` }}
                  style={styles.modalImage}
                />
                <Text style={styles.modalTitle}>診断内容</Text>
                <Text style={styles.modalText}>{activeEntry.diagnosis}</Text>
                <Text style={styles.modalMeta}>
                  {formatDateTime(activeEntry.timestamp)} ・ {activeEntry.processing_time}s
                </Text>

                <View style={styles.chatSection}>
                  <Text style={styles.modalTitle}>チャット</Text>
                  {(activeEntry.chat || []).length === 0 ? (
                    <Text style={styles.muted}>まだチャットがありません。</Text>
                  ) : (
                    (activeEntry.chat || []).map((chat, index) => (
                      <View
                        key={`${chat.timestamp}-${index}`}
                        style={[
                          styles.chatBubble,
                          chat.role === "user" ? styles.chatUser : styles.chatAssistant
                        ]}
                      >
                        <Text style={styles.chatText}>{chat.message}</Text>
                        <Text style={styles.chatMeta}>{formatDateTime(chat.timestamp)}</Text>
                      </View>
                    ))
                  )}
                  <TextInput
                    style={styles.chatInput}
                    placeholder="追加で質問..."
                    value={chatInput}
                    onChangeText={setChatInput}
                    multiline
                  />
                  <Pressable
                    style={[styles.primaryButton, sendingChat && styles.disabledButton]}
                    onPress={sendChat}
                    disabled={sendingChat}
                  >
                    <Text style={styles.primaryButtonText}>
                      {sendingChat ? "送信中..." : "送信"}
                    </Text>
                  </Pressable>
                </View>
              </>
            ) : null}
          </ScrollView>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f3ef"
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40
  },
  hero: {
    marginBottom: 24
  },
  kicker: {
    textTransform: "uppercase",
    letterSpacing: 2,
    color: "#35655a",
    fontSize: 12,
    fontWeight: "600"
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#1c2a24",
    marginTop: 8
  },
  subtitle: {
    color: "#55605c",
    marginTop: 8
  },
  heroMeta: {
    flexDirection: "row",
    marginTop: 16,
    flexWrap: "wrap"
  },
  badge: {
    backgroundColor: "#ffffff",
    padding: 12,
    borderRadius: 16,
    marginRight: 12,
    marginBottom: 12,
    flex: 1,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 }
  },
  badgeTitle: {
    fontSize: 12,
    color: "#7a8a83"
  },
  badgeText: {
    fontWeight: "600",
    marginTop: 4
  },
  card: {
    backgroundColor: "#ffffff",
    padding: 20,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 }
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 6
  },
  cardSub: {
    color: "#55605c",
    marginBottom: 12
  },
  primaryButton: {
    backgroundColor: "#2d7d6a",
    paddingVertical: 12,
    borderRadius: 999,
    alignItems: "center"
  },
  primaryButtonText: {
    color: "#fff",
    fontWeight: "600"
  },
  disabledButton: {
    opacity: 0.6
  },
  helperText: {
    marginTop: 10,
    fontSize: 12,
    color: "#7a8a83"
  },
  section: {
    marginTop: 24
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600"
  },
  sectionMeta: {
    fontSize: 12,
    color: "#7a8a83"
  },
  muted: {
    color: "#7a8a83"
  },
  resultCard: {
    flexDirection: "row",
    backgroundColor: "#fff",
    padding: 12,
    borderRadius: 16,
    marginBottom: 12
  },
  resultImage: {
    width: 84,
    height: 84,
    borderRadius: 12,
    marginRight: 12
  },
  resultBody: {
    flex: 1
  },
  resultMeta: {
    fontSize: 11,
    color: "#7a8a83",
    marginBottom: 4
  },
  resultText: {
    color: "#1c2a24"
  },
  historyItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    padding: 12,
    borderRadius: 16,
    marginBottom: 10
  },
  historyThumb: {
    width: 64,
    height: 64,
    borderRadius: 12,
    marginRight: 12
  },
  historyBody: {
    flex: 1
  },
  linkText: {
    color: "#2d7d6a",
    fontWeight: "600"
  },
  modalContainer: {
    flex: 1,
    backgroundColor: "#f5f3ef"
  },
  modalContent: {
    padding: 20
  },
  modalImage: {
    width: "100%",
    height: 280,
    borderRadius: 20,
    marginBottom: 16
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 6
  },
  modalText: {
    color: "#1c2a24",
    marginBottom: 10
  },
  modalMeta: {
    color: "#7a8a83",
    marginBottom: 16
  },
  chatSection: {
    marginTop: 12
  },
  chatBubble: {
    padding: 12,
    borderRadius: 16,
    marginBottom: 10
  },
  chatUser: {
    backgroundColor: "#d9ece6",
    alignSelf: "flex-end"
  },
  chatAssistant: {
    backgroundColor: "#ffffff",
    alignSelf: "flex-start"
  },
  chatText: {
    color: "#1c2a24"
  },
  chatMeta: {
    fontSize: 10,
    color: "#7a8a83",
    marginTop: 4
  },
  chatInput: {
    borderWidth: 1,
    borderColor: "#d9e0da",
    borderRadius: 12,
    padding: 10,
    marginBottom: 10,
    minHeight: 60
  }
});
